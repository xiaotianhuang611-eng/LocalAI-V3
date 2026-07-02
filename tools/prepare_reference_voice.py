from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_PATH = DATA_DIR / "reference.wav"

TARGET_SAMPLE_RATE = 24000
TARGET_PEAK = 0.85


def safe_normalize(audio: np.ndarray, peak: float = TARGET_PEAK) -> np.ndarray:
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


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1).astype(np.float32)


def simple_resample(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr:
        return audio.astype(np.float32)

    if audio.size == 0:
        return audio.astype(np.float32)

    old_len = len(audio)
    new_len = int(round(old_len * new_sr / old_sr))

    if new_len <= 1:
        return audio.astype(np.float32)

    old_x = np.linspace(0.0, 1.0, old_len)
    new_x = np.linspace(0.0, 1.0, new_len)

    resampled = np.interp(new_x, old_x, audio)

    return resampled.astype(np.float32)


def trim_audio(audio: np.ndarray, sample_rate: int, start: float, duration: float | None) -> np.ndarray:
    start_index = max(0, int(start * sample_rate))

    if duration is None or duration <= 0:
        return audio[start_index:]

    end_index = min(len(audio), start_index + int(duration * sample_rate))

    return audio[start_index:end_index]


def run_ffmpeg_convert(input_path: Path, output_path: Path, start: float, duration: float | None) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg first, or use a WAV input file."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_filter_parts = []

    if start > 0 or (duration is not None and duration > 0):
        if duration is not None and duration > 0:
            audio_filter_parts.append(f"atrim=start={start}:duration={duration}")
        else:
            audio_filter_parts.append(f"atrim=start={start}")
        audio_filter_parts.append("asetpts=PTS-STARTPTS")

    audio_filter_parts.extend(
        [
            "highpass=f=70",
            "lowpass=f=8000",
            "loudnorm=I=-20:TP=-3:LRA=11",
        ]
    )

    audio_filter = ",".join(audio_filter_parts)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-af",
        audio_filter,
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    print("[ReferencePrep] Running ffmpeg:")
    print(" ".join(cmd))

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed.\n\n"
            f"STDERR:\n{completed.stderr}"
        )


def load_wav_without_ffmpeg(input_path: Path, start: float, duration: float | None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(input_path), dtype="float32")

    audio = to_mono(audio)
    audio = trim_audio(audio, sample_rate, start, duration)
    audio = simple_resample(audio, sample_rate, TARGET_SAMPLE_RATE)

    return audio, TARGET_SAMPLE_RATE


def prepare_reference_voice(
    input_path: Path,
    output_path: Path = OUTPUT_PATH,
    start: float = 0.0,
    duration: float | None = 10.0,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    temp_wav = TEMP_DIR / "reference_converted_temp.wav"

    suffix = input_path.suffix.lower()

    if shutil.which("ffmpeg") is not None:
        run_ffmpeg_convert(
            input_path=input_path,
            output_path=temp_wav,
            start=start,
            duration=duration,
        )

        audio, sample_rate = sf.read(str(temp_wav), dtype="float32")
        audio = to_mono(audio)

    else:
        if suffix != ".wav":
            raise RuntimeError(
                "Your input is not WAV and ffmpeg is not installed.\n"
                "Please install ffmpeg, or send/convert the file as WAV first."
            )

        print("[ReferencePrep] ffmpeg not found. Using soundfile WAV path.")
        audio, sample_rate = load_wav_without_ffmpeg(
            input_path=input_path,
            start=start,
            duration=duration,
        )

    audio = to_mono(audio)
    audio = safe_normalize(audio)

    audio_seconds = len(audio) / float(TARGET_SAMPLE_RATE)

    if audio_seconds < 3:
        print(
            f"[ReferencePrep][WARN] Reference is only {audio_seconds:.2f}s. "
            "XTTS may work poorly. Recommended: 7-12s."
        )

    if audio_seconds > 20:
        print(
            f"[ReferencePrep][WARN] Reference is {audio_seconds:.2f}s. "
            "Recommended: 7-12s. Consider using --duration 10."
        )

    sf.write(str(output_path), audio, TARGET_SAMPLE_RATE)

    print("")
    print("[ReferencePrep] Done.")
    print(f"[ReferencePrep] Input:  {input_path}")
    print(f"[ReferencePrep] Output: {output_path}")
    print(f"[ReferencePrep] Format: WAV, mono, {TARGET_SAMPLE_RATE} Hz")
    print(f"[ReferencePrep] Length: {audio_seconds:.2f}s")
    print("")
    print("Now XTTS will use this file automatically:")
    print(output_path)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert any common voice recording into LocalAI_V3 XTTS reference.wav"
    )

    parser.add_argument(
        "input",
        help="Input audio path, e.g. C:\\Users\\111\\Desktop\\voice.m4a",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds. Default: 0",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds. Default: 10. Use 7-12 seconds normally.",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output WAV path. Default: data/reference.wav",
    )

    args = parser.parse_args()

    prepare_reference_voice(
        input_path=Path(args.input),
        output_path=Path(args.output),
        start=float(args.start),
        duration=float(args.duration),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())