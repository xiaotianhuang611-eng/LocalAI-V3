from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
REFERENCE_WAV = DATA_DIR / "reference.wav"
OUTPUT_WAV = DATA_DIR / "output.wav"

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def apply_xtts_patches() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Hide torchcodec
    for name in list(sys.modules.keys()):
        if name == "torchcodec" or name.startswith("torchcodec."):
            del sys.modules[name]

    if not hasattr(importlib.util.find_spec, "_xtts_no_torchcodec"):
        original_find_spec = importlib.util.find_spec

        def find_spec_no_torchcodec(name, package=None):
            if name == "torchcodec" or name.startswith("torchcodec."):
                return None
            return original_find_spec(name, package)

        find_spec_no_torchcodec._xtts_no_torchcodec = True
        importlib.util.find_spec = find_spec_no_torchcodec

    # Patch torchaudio.load/info
    try:
        import torch
        import torchaudio

        def _soundfile_load(
            filepath,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format=None,
            buffer_size: int = 4096,
            backend=None,
        ):
            data, sample_rate = sf.read(str(filepath), dtype="float32", always_2d=True)

            if frame_offset:
                data = data[frame_offset:]

            if num_frames is not None and num_frames > 0:
                data = data[:num_frames]

            tensor = torch.from_numpy(data)

            if channels_first:
                tensor = tensor.transpose(0, 1)

            return tensor, sample_rate

        def _soundfile_info(filepath, format=None, buffer_size: int = 4096, backend=None):
            info = sf.info(str(filepath))

            class AudioMetaData:
                def __init__(
                    self,
                    sample_rate,
                    num_frames,
                    num_channels,
                    bits_per_sample,
                    encoding,
                ):
                    self.sample_rate = sample_rate
                    self.num_frames = num_frames
                    self.num_channels = num_channels
                    self.bits_per_sample = bits_per_sample
                    self.encoding = encoding

            return AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=16,
                encoding=info.format,
            )

        torchaudio.load = _soundfile_load
        torchaudio.info = _soundfile_info

    except Exception:
        pass

    # Patch torch.load weights_only
    try:
        import torch

        real_torch_load = (
            torch.load.__wrapped__
            if hasattr(torch.load, "__wrapped__")
            else torch.load
        )

        try:
            supports_weights_only = "weights_only" in inspect.signature(real_torch_load).parameters
        except Exception:
            supports_weights_only = False

        def safe_torch_load(*args, **kwargs):
            if supports_weights_only:
                kwargs.setdefault("weights_only", False)
            else:
                kwargs.pop("weights_only", None)

            return real_torch_load(*args, **kwargs)

        safe_torch_load.__wrapped__ = real_torch_load
        torch.load = safe_torch_load

    except Exception:
        pass

    # Patch Coqpit
    try:
        from coqpit import Coqpit
        import coqpit.coqpit as coqpit_module

        if hasattr(coqpit_module, "_deserialize"):
            original_deserialize_func = coqpit_module._deserialize

            def lenient_deserialize_func(value, field_type):
                try:
                    return original_deserialize_func(value, field_type)
                except (TypeError, ValueError) as exc:
                    msg = str(exc).lower()

                    if (
                        "does not match" in msg
                        or "field type" in msg
                        or "value_error" in msg
                    ):
                        return value

                    raise

            coqpit_module._deserialize = lenient_deserialize_func

        original_deserialize = Coqpit.deserialize

        def lenient_deserialize(self, data):
            try:
                return original_deserialize(self, data)
            except (TypeError, ValueError) as exc:
                msg = str(exc).lower()

                if (
                    "does not match" in msg
                    or "field type" in msg
                    or "value_error" in msg
                ):
                    for key, value in data.items():
                        try:
                            setattr(self, key, value)
                        except Exception:
                            pass

                    if hasattr(self, "__dict__"):
                        self.__dict__.update(data)

                    return self

                raise

        Coqpit.deserialize = lenient_deserialize

    except Exception:
        pass

    # Patch pydantic
    try:
        from pydantic import BaseModel

        if not hasattr(BaseModel.__init__, "_xtts_patched"):
            original_init = BaseModel.__init__

            def lenient_init(self, **data):
                try:
                    original_init(self, **data)
                except Exception as exc:
                    msg = str(exc).lower()

                    if (
                        "does not match" in msg
                        or "field type" in msg
                        or "value_error" in msg
                    ):
                        self.__dict__.update(data)

                        if hasattr(self, "__pydantic_fields_set__"):
                            object.__setattr__(
                                self,
                                "__pydantic_fields_set__",
                                set(data.keys()),
                            )
                    else:
                        raise

            lenient_init._xtts_patched = True
            BaseModel.__init__ = lenient_init

    except Exception:
        pass


def record_reference() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("[XTTS TEST] Recording reference voice for 8 seconds...")
    print("[XTTS TEST] Speak English clearly now.")

    sample_rate = 16000
    duration = 8

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    sf.write(
        str(REFERENCE_WAV),
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    print(f"[XTTS TEST] Reference saved: {REFERENCE_WAV}")


def main() -> int:
    print("[XTTS TEST] Script started.")

    apply_xtts_patches()

    import torch
    from TTS.api import TTS

    print("[XTTS TEST] torch:", torch.__version__)
    print("[XTTS TEST] cuda:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("[XTTS TEST] gpu:", torch.cuda.get_device_name(0))

    if not REFERENCE_WAV.exists():
        record_reference()
    else:
        print(f"[XTTS TEST] Existing reference found: {REFERENCE_WAV}")

    print("[XTTS TEST] Loading XTTS...")

    start_load = time.perf_counter()

    tts = TTS(
        XTTS_MODEL,
        gpu=torch.cuda.is_available(),
        progress_bar=False,
    )

    if torch.cuda.is_available():
        tts = tts.to("cuda")

    load_time = time.perf_counter() - start_load

    print(f"[XTTS TEST] Loaded in {load_time:.2f}s")

    try:
        model = tts.synthesizer.tts_model
        print("[XTTS TEST] model device:", next(model.parameters()).device)
    except Exception as exc:
        print("[XTTS TEST] model device check failed:", exc)

    wav_data, wav_sr = sf.read(
        str(REFERENCE_WAV),
        dtype="float32",
    )

    tmp_ref = TEMP_DIR / "_tmp_ref_pcm.wav"

    sf.write(
        str(tmp_ref),
        wav_data,
        wav_sr,
        subtype="PCM_16",
    )

    text = "Hello, I am your local AI assistant running on this laptop."

    print(f"[XTTS TEST] Synthesizing: {text!r}")

    start = time.perf_counter()

    with torch.inference_mode():
        audio_array = tts.tts(
            text=text,
            speaker_wav=str(tmp_ref),
            language="en",
            enable_text_splitting=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    audio_np = np.asarray(audio_array, dtype=np.float32)
    sample_rate = int(tts.synthesizer.output_sample_rate)

    duration = len(audio_np) / float(sample_rate)
    rtf = elapsed / duration if duration > 0 else 0.0

    print(
        f"[XTTS TEST] Synthesized in {elapsed:.2f}s | "
        f"audio={duration:.2f}s | RTF={rtf:.3f}"
    )

    print("[XTTS TEST] Playing audio...")

    sd.play(audio_np, sample_rate)
    sd.wait()

    sf.write(
        str(OUTPUT_WAV),
        audio_np,
        sample_rate,
    )

    print(f"[XTTS TEST] Saved: {OUTPUT_WAV}")
    print("[XTTS TEST] Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())