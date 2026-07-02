from __future__ import annotations

import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


class FasterWhisperASR:
    def __init__(
        self,
        root_dir: Path,
        model_name: str = "small.en",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_name = model_name

        self.data_dir = self.root_dir / "data"
        self.temp_dir = self.data_dir / "temp"
        self.question_wav = self.temp_dir / "question.wav"

        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

    def load(self) -> None:
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ASR] Loading faster-whisper model: {self.model_name}")

        start = time.perf_counter()

        # Prefer CUDA first.
        try:
            self.model = WhisperModel(
                self.model_name,
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
                num_workers=1,
            )
            self.device = "cuda"
            self.compute_type = "float16"

        except Exception as exc:
            print(f"[ASR] CUDA load failed, fallback to CPU: {exc}")

            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=8,
                num_workers=1,
            )
            self.device = "cpu"
            self.compute_type = "int8"

        elapsed = time.perf_counter() - start

        print(
            f"[ASR] Loaded in {elapsed:.2f}s | "
            f"device={self.device} | compute_type={self.compute_type}"
        )

    def record_question(
        self,
        seconds: int = 5,
        sample_rate: int = 16000,
    ) -> Path:
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ASR] Recording question for {seconds}s...")
        print("[ASR] Speak English clearly now.")

        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )

        sd.wait()

        sf.write(
            str(self.question_wav),
            audio,
            sample_rate,
            subtype="PCM_16",
        )

        print(f"[ASR] Saved question audio: {self.question_wav}")

        return self.question_wav

    def transcribe_file(self, wav_path: Path) -> str:
        if self.model is None:
            raise RuntimeError("ASR model is not loaded. Call load() first.")

        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        print(f"[ASR] Transcribing: {wav_path}")

        start = time.perf_counter()

        segments, info = self.model.transcribe(
            str(wav_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )

        texts: list[str] = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)

        elapsed = time.perf_counter() - start

        result = " ".join(texts).strip()

        print(
            f"[ASR] Transcribed in {elapsed:.2f}s | "
            f"language={info.language} | probability={info.language_probability:.2f}"
        )

        print(f"[ASR] Text: {result!r}")

        return result

    def record_and_transcribe(self, seconds: int = 5) -> str:
        wav_path = self.record_question(seconds=seconds)
        return self.transcribe_file(wav_path)