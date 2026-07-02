from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime


def main() -> int:
    runtime = ModelRuntime(root_dir=ROOT_DIR)

    runtime.load()

    print("[TEST VOICE] Now loading ASR...")
    runtime.load_asr()

    print("[TEST VOICE] Speak your question after the recording starts.")
    print("[TEST VOICE] Example: What can you do for me?")

    user_text, response = runtime.listen_ask_and_speak(seconds=5)

    print("[TEST VOICE] User said:", user_text)
    print("[TEST VOICE] Gemma replied:", response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())