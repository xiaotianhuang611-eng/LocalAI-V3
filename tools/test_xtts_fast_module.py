from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.xtts_fast import XTTSFast


def main() -> int:
    xtts = XTTSFast(root_dir=ROOT_DIR)
    xtts.load()
    xtts.speak("Hello, I am your local AI assistant running on this laptop.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())