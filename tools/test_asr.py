from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.asr import FasterWhisperASR


def main() -> int:
    asr = FasterWhisperASR(root_dir=ROOT_DIR, model_name="small.en")
    asr.load()

    text = asr.record_and_transcribe(seconds=5)

    print("[TEST ASR] Final text:", text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())