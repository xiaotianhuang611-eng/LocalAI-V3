from __future__ import annotations

import gc
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime
from core.qwen_vision import QwenVision


def release_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    time.sleep(1.0)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            r"  .\.venv\Scripts\python.exe tools\test_vision_runtime.py C:\path\to\image.jpg"
        )
        return 1

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"[TEST VISION RUNTIME] Image not found: {image_path}")
        return 1

    print("[TEST VISION RUNTIME] Phase 1: Qwen-VL image analysis")
    print("[TEST VISION RUNTIME] Image:", image_path)

    qwen = QwenVision(root_dir=ROOT_DIR)

    try:
        qwen.load()

        vision_answer = qwen.describe_image(
            image_path=image_path,
            question="What is in this image? Describe it clearly and concisely.",
        )

    finally:
        qwen.unload()
        del qwen
        release_memory()

    print()
    print("[TEST VISION RUNTIME] Qwen-VL answer:")
    print(vision_answer)
    print()

    print("[TEST VISION RUNTIME] Phase 2: Gemma + XTTSFast")
    print("[TEST VISION RUNTIME] Loading Gemma and XTTSFast after Qwen-VL unload...")

    runtime = ModelRuntime(root_dir=ROOT_DIR)
    runtime.load()

    print("[TEST VISION RUNTIME] Rewriting Qwen-VL answer for voice...")

    spoken_answer = runtime.gemma.rewrite_for_voice(vision_answer)

    print()
    print("[TEST VISION RUNTIME] Spoken answer:")
    print(spoken_answer)
    print()

    print("[TEST VISION RUNTIME] Speaking...")
    runtime.speak(spoken_answer)

    print("[TEST VISION RUNTIME] Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())