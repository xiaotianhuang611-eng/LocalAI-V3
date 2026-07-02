from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime


def main() -> int:
    runtime = ModelRuntime(root_dir=ROOT_DIR)
    runtime.load()

    text = "Hello, what can you do?"

    print("[TEST] User:", text)

    response = runtime.ask(text)

    print("[TEST] AI:", response)

    runtime.speak(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())