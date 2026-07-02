from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.qwen_vision import QwenVision


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print(r"  .\.venv\Scripts\python.exe tools\test_qwen_vision.py C:\path\to\image.jpg")
        return 1

    image_path = Path(sys.argv[1])

    qwen = QwenVision(root_dir=ROOT_DIR)

    try:
        qwen.load()

        answer = qwen.describe_image(
            image_path=image_path,
            question="What is in this image?",
        )

        print("[TEST QWEN VISION] Answer:")
        print(answer)

    finally:
        qwen.unload()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())