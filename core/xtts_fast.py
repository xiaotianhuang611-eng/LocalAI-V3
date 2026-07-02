from __future__ import annotations

import gc
import importlib.util
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf


_PATCHED = False
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patch_runtime_for_xtts() -> None:
    """
    Runtime compatibility patches for Coqui XTTS on local Windows GPU setup.

    This keeps the old stable behavior:
    - avoids torchcodec import path problems
    - makes torch.load compatible with old TTS checkpoints
    - makes torchaudio load/info use soundfile
    - relaxes some config deserialization edge cases
    """
    global _PATCHED

    if _PATCHED:
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Hide torchcodec because some torchaudio / TTS paths try to detect it.
    def safe_find_spec(name: str, package: str | None = None):
        if name == "torchcodec" or name.startswith("torchcodec."):
            return None
        return _ORIGINAL_FIND_SPEC(name, package)

    importlib.util.find_spec = safe_find_spec

    for module_name in list(sys.modules.keys()):
        if module_name == "torchcodec" or module_name.startswith("torchcodec."):
            sys.modules.pop(module_name, None)

    # Patch torch.load so older Coqui checkpoints can load on newer PyTorch.
    try:
        import torch

        if not getattr(torch.load, "_xtts_patched", False):
            original_torch_load = torch.load

            def patched_torch_load(*args: Any, **kwargs: Any):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            patched_torch_load._xtts_patched = True  # type: ignore[attr-defined]
            torch.load = patched_torch_load  # type: ignore[assignment]

    except Exception:
        pass

    # Patch torchaudio.load/info to soundfile to avoid backend issues.
    try:
        import torchaudio
        import torch

        def patched_torchaudio_load(filepath: str | Path, *args: Any, **kwargs: Any):
            data, sample_rate = sf.read(str(filepath), dtype="float32", always_2d=True)
            data = data.T
            tensor = torch.from_numpy(data)
            return tensor, sample_rate

        def patched_torchaudio_info(filepath: str | Path, *args: Any, **kwargs: Any):
            info = sf.info(str(filepath))

            class AudioMetaData:
                def __init__(self) -> None:
                    self.sample_rate = info.samplerate
                    self.num_frames = info.frames
                    self.num_channels = info.channels
                    self.bits_per_sample = 16
                    self.encoding = "PCM_S"

                def __repr__(self) -> str:
                    return (
                        "AudioMetaData("
                        f"sample_rate={self.sample_rate}, "
                        f"num_frames={self.num_frames}, "
                        f"num_channels={self.num_channels}, "
                        f"bits_per_sample={self.bits_per_sample}, "
                        f"encoding={self.encoding!r})"
                    )

            return AudioMetaData()

        torchaudio.load = patched_torchaudio_load  # type: ignore[assignment]
        torchaudio.info = patched_torchaudio_info  # type: ignore[assignment]

    except Exception:
        pass

    # Lenient Coqpit deserialize patch.
    try:
        import coqpit

        Coqpit = getattr(coqpit, "Coqpit", None)

        if Coqpit is not None and hasattr(Coqpit, "deserialize"):
            original_deserialize = Coqpit.deserialize

            if not getattr(original_deserialize, "_xtts_patched", False):

                def patched_deserialize(self, data: Any) -> Any:
                    try:
                        return original_deserialize(self, data)
                    except TypeError:
                        if isinstance(data, dict):
                            for key, value in data.items():
                                try:
                                    setattr(self, key, value)
                                except Exception:
                                    pass
                        return self

                patched_deserialize._xtts_patched = True  # type: ignore[attr-defined]
                Coqpit.deserialize = patched_deserialize

    except Exception:
        pass

    # Lenient pydantic init patch for dependency edge cases.
    try:
        from pydantic import BaseModel

        original_init = BaseModel.__init__

        if not getattr(original_init, "_xtts_patched", False):

            def patched_base_model_init(self, **data: Any) -> None:
                try:
                    original_init(self, **data)
                except Exception:
                    for key, value in data.items():
                        try:
                            object.__setattr__(self, key, value)
                        except Exception:
                            pass

            patched_base_model_init._xtts_patched = True  # type: ignore[attr-defined]
            BaseModel.__init__ = patched_base_model_init  # type: ignore[assignment]

    except Exception:
        pass

    _PATCHED = True


class XTTSFast:
    def __init__(
        self,
        root_dir: Path,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "en",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_name = model_name
        self.language = language

        self.data_dir = self.root_dir / "data"
        self.temp_dir = self.data_dir / "temp"
        self.reference_path = self.data_dir / "reference.wav"
        self.output_path = self.data_dir / "output.wav"
        self.prepared_reference_path = self.temp_dir / "reference_prepared.wav"

        self.tts = None
        self.sample_rate = 24000

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        if self.tts is not None:
            return

        _patch_runtime_for_xtts()

        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[XTTSFast] Loading XTTS model: {self.model_name}")
        print(f"[XTTSFast] Device: {device}")

        start = time.perf_counter()

        try:
            self.tts = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=(device == "cuda"),
            )
        except TypeError:
            self.tts = TTS(
                model_name=self.model_name,
                progress_bar=False,
            )
            if hasattr(self.tts, "to"):
                self.tts.to(device)

        try:
            if hasattr(self.tts, "synthesizer"):
                self.sample_rate = int(getattr(self.tts.synthesizer, "output_sample_rate", 24000))
        except Exception:
            self.sample_rate = 24000

        elapsed = time.perf_counter() - start
        print(f"[XTTSFast] Loaded in {elapsed:.2f}s | sample_rate={self.sample_rate}")

    def record_reference(self, seconds: int = 8, sample_rate: int = 24000) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print(f"[XTTSFast] Recording reference voice for {seconds}s...")
        print("[XTTSFast] Speak clearly and naturally.")

        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = self._safe_normalize(audio)

        sf.write(str(self.reference_path), audio, sample_rate)

        print(f"[XTTSFast] Reference saved: {self.reference_path}")
        return self.reference_path

    def speak(self, text: str) -> None:
        if self.tts is None:
            self.load()

        original_text = str(text).strip()
        text = self._sanitize_for_tts(original_text)

        if not text:
            print("[XTTSFast] No speakable text after sanitization.")
            return

        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference voice not found: {self.reference_path}\n"
                "Please click Record Reference first."
            )

        speaker_wav = self._prepare_temp_reference()

        print(f"[XTTSFast] Speaking sanitized text: {text[:180]}")

        start = time.perf_counter()

        try:
            wav = self.tts.tts(
                text=text,
                speaker_wav=str(speaker_wav),
                language=self.language,
                enable_text_splitting=False,
            )
        except TypeError:
            wav = self.tts.tts(
                text=text,
                speaker_wav=str(speaker_wav),
                language=self.language,
                split_sentences=False,
            )

        audio = self._to_numpy_audio(wav)
        audio = self._safe_normalize(audio)

        sf.write(str(self.output_path), audio, self.sample_rate)

        synth_time = time.perf_counter() - start
        audio_seconds = len(audio) / float(self.sample_rate)
        rtf = synth_time / audio_seconds if audio_seconds > 0 else 0.0

        print(
            f"[XTTSFast] Synthesized in {synth_time:.2f}s | "
            f"audio={audio_seconds:.2f}s | RTF={rtf:.3f}"
        )
        print(f"[XTTSFast] Output saved: {self.output_path}")

        sd.play(audio, self.sample_rate)
        sd.wait()

    def _sanitize_for_tts(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)

        # Remove hidden/control characters.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

        # Remove Markdown code blocks.
        text = re.sub(r"```[\s\S]*?```", " code omitted. ", text)

        # Remove inline code marks.
        text = text.replace("`", "")

        # Replace URLs and emails with readable placeholders.
        text = re.sub(r"https?://\S+", " link ", text)
        text = re.sub(r"www\.\S+", " link ", text)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            " email address ",
            text,
        )

        # Remove common Markdown symbols.
        text = re.sub(r"[*_~#]+", " ", text)

        # Convert block quotes and bullets into pauses.
        text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-•●▪▫]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n\s*[-•●▪▫]\s*", ". ", text)

        # Normalize line breaks into sentence pauses.
        text = re.sub(r"\n{2,}", ". ", text)
        text = text.replace("\n", ". ")

        # Replace common technical symbols.
        replacements = [
            ("=>", " to "),
            ("->", " to "),
            ("→", " to "),
            ("←", " from "),
            ("<=", " less than or equal to "),
            (">=", " greater than or equal to "),
            ("!=", " not equal to "),
            ("==", " equals "),
            ("=", " equals "),
            ("+", " plus "),
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("℃", " degrees Celsius "),
            ("°C", " degrees Celsius "),
            ("°F", " degrees Fahrenheit "),
            ("￥", " yuan "),
            ("$", " dollars "),
            ("€", " euros "),
            ("£", " pounds "),
            ("|", " "),
            ("\\", " "),
            ("/", " "),
            ("{", " "),
            ("}", " "),
            ("[", " "),
            ("]", " "),
            ("(", " "),
            (")", " "),
            ("<", " "),
            (">", " "),
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        # Remove emojis and rare Unicode symbols.
        text = re.sub(
            r"[\U00010000-\U0010ffff]",
            " ",
            text,
            flags=re.UNICODE,
        )

        # Keep English, Chinese, numbers, and common punctuation.
        text = re.sub(
            r"[^A-Za-z0-9\u4e00-\u9fff.,!?;:'：，。！？；、 -]+",
            " ",
            text,
        )

        # Reduce repeated punctuation.
        text = re.sub(r"[!！]{2,}", "!", text)
        text = re.sub(r"[?？]{2,}", "?", text)
        text = re.sub(r"[.。]{3,}", ".", text)
        text = re.sub(r"[,，]{2,}", ",", text)

        # Collapse spaces.
        text = re.sub(r"\s+", " ", text).strip()

        # Avoid sending extremely long text to XTTS at once.
        max_chars = 900

        if len(text) > max_chars:
            cut = text[:max_chars]

            last_stop = max(
                cut.rfind("."),
                cut.rfind("!"),
                cut.rfind("?"),
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
            )

            if last_stop > 200:
                text = cut[: last_stop + 1]
            else:
                text = cut.rsplit(" ", 1)[0].strip() + "."

        return text

    def _prepare_temp_reference(self) -> Path:
        data, sample_rate = sf.read(str(self.reference_path), dtype="float32")

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        data = np.asarray(data, dtype=np.float32)
        data = self._safe_normalize(data)

        sf.write(str(self.prepared_reference_path), data, sample_rate)

        return self.prepared_reference_path

    def _safe_normalize(self, audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            return audio

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        max_abs = float(np.max(np.abs(audio)))

        if max_abs < 1e-8:
            return audio

        audio = audio / max_abs * peak
        audio = np.clip(audio, -1.0, 1.0)

        return audio.astype(np.float32)

    def _to_numpy_audio(self, wav: Any) -> np.ndarray:
        try:
            import torch

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
        except Exception:
            pass

        if isinstance(wav, list):
            wav = np.asarray(wav, dtype=np.float32)

        if isinstance(wav, tuple):
            wav = np.asarray(wav[0], dtype=np.float32)

        wav = np.asarray(wav, dtype=np.float32)

        if wav.ndim > 1:
            wav = np.squeeze(wav)

        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        return wav.astype(np.float32)

    def unload(self) -> None:
        self.tts = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass