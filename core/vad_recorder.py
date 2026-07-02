from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class EnergyVADRecorder:
    """
    Lightweight local VAD recorder.

    No extra dependency.
    Uses microphone RMS energy to detect:
    - speech start
    - speech end
    """

    def __init__(
        self,
        root_dir: Path,
        sample_rate: int = 16000,
        block_ms: int = 30,
        start_threshold: float = 0.012,
        end_threshold: float = 0.008,
        silence_end_seconds: float = 2.0,
        pre_roll_seconds: float = 0.35,
        min_record_seconds: float = 1.5,
        max_record_seconds: float = 30.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.block_ms = block_ms

        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.silence_end_seconds = silence_end_seconds
        self.pre_roll_seconds = pre_roll_seconds
        self.min_record_seconds = min_record_seconds
        self.max_record_seconds = max_record_seconds

        self.temp_dir = self.root_dir / "data" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.output_path = self.temp_dir / "question_auto.wav"

    def record_auto_stop(self, output_path: Path | None = None) -> Path:
        output_path = Path(output_path) if output_path else self.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        block_size = int(self.sample_rate * self.block_ms / 1000)
        pre_roll_blocks = max(1, int(self.pre_roll_seconds / (self.block_ms / 1000)))
        silence_end_blocks = max(1, int(self.silence_end_seconds / (self.block_ms / 1000)))

        pre_roll_buffer: list[np.ndarray] = []
        recorded_blocks: list[np.ndarray] = []

        started = False
        silent_blocks = 0

        start_time = time.perf_counter()
        speech_start_time: float | None = None

        print("[VAD] Auto recording started.")
        print("[VAD] Speak now. Recording will stop after silence.")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:
            while True:
                block, overflowed = stream.read(block_size)

                if overflowed:
                    print("[VAD][WARN] Audio overflow detected.")

                audio = np.asarray(block, dtype=np.float32).reshape(-1)

                rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))

                elapsed = time.perf_counter() - start_time

                if not started:
                    pre_roll_buffer.append(audio)

                    if len(pre_roll_buffer) > pre_roll_blocks:
                        pre_roll_buffer.pop(0)

                    if rms >= self.start_threshold:
                        started = True
                        speech_start_time = time.perf_counter()

                        recorded_blocks.extend(pre_roll_buffer)
                        recorded_blocks.append(audio)

                        print(f"[VAD] Speech detected. rms={rms:.5f}")

                else:
                    recorded_blocks.append(audio)

                    speech_elapsed = (
                        time.perf_counter() - speech_start_time
                        if speech_start_time is not None
                        else 0.0
                    )

                    if rms < self.end_threshold:
                        silent_blocks += 1
                    else:
                        silent_blocks = 0

                    enough_speech = speech_elapsed >= self.min_record_seconds
                    enough_silence = silent_blocks >= silence_end_blocks

                    if enough_speech and enough_silence:
                        print("[VAD] Silence detected. Stop recording.")
                        break

                if elapsed >= self.max_record_seconds:
                    print("[VAD] Max recording time reached. Stop recording.")
                    break

        if not recorded_blocks:
            print("[VAD][WARN] No speech detected. Saving short silence.")
            audio_out = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        else:
            audio_out = np.concatenate(recorded_blocks).astype(np.float32)

        audio_out = self._safe_normalize(audio_out)

        sf.write(str(output_path), audio_out, self.sample_rate)

        seconds = len(audio_out) / float(self.sample_rate)

        print(f"[VAD] Saved: {output_path}")
        print(f"[VAD] Duration: {seconds:.2f}s")

        return output_path

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