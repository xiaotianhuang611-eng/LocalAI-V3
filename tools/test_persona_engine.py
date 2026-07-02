from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.model_runtime import ModelRuntime


def main() -> int:
    runtime = ModelRuntime(
        root_dir=ROOT_DIR,
        personality_mode="project_engineer",
        emotion_enabled=True,
    )

    runtime.load()

    tests = [
        ("project_engineer", "The program has an error. Tell me how to debug it."),
        ("cute_companion", "I finally made the voice assistant fast. Say something short."),
        ("study_tutor", "Explain what RAG means in simple terms."),
        ("thesis_assistant", "Explain our project architecture academically."),
        ("benchmark_analyst", "Explain what latency and RTF mean for our XTTS benchmark."),
    ]

    for mode, prompt in tests:
        print("\n" + "=" * 70)
        print(f"[TEST MODE] {mode}")
        print("=" * 70)

        runtime.set_personality_mode(mode)

        response = runtime.ask(prompt)

        print("")
        print("[USER]")
        print(prompt)
        print("")
        print("[AI]")
        print(response)
        print("")
        print("[EMOTION]")
        print(runtime.get_emotion_summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())