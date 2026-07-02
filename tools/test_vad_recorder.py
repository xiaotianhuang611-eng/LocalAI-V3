from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.asr import FasterWhisperASR
from core.vad_recorder import EnergyVADRecorder


def main() -> int:
    recorder = EnergyVADRecorder(
        root_dir=ROOT_DIR,
        start_threshold=0.012,
        end_threshold=0.008,
        silence_end_seconds=1.5,
        max_record_seconds=30.0,
    )

    wav_path = recorder.record_auto_stop()

    print("")
    print("[TEST] Loading ASR...")

    asr = FasterWhisperASR(root_dir=ROOT_DIR, model_name="small.en")
    asr.load()

    print("[TEST] Transcribing VAD recording...")

    text = asr.transcribe_file(wav_path)

    print("")
    print("=" * 50)
    print("[VAD + ASR RESULT]")
    print(text)
    print("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())