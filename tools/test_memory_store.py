from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.memory_store import MemoryStore


def main() -> int:
    memory = MemoryStore(root_dir=ROOT_DIR)

    print("=" * 60)
    print("[MEMORY SUMMARY]")
    print(memory.short_summary())

    print("")
    print("=" * 60)
    print("[MEMORY CONTEXT]")
    print(memory.retrieve_context("What is our LocalAI project path and current version?"))

    memory.add_stable_fact("Memory test completed successfully for V3.3.")
    memory.add_event("test", "The local memory store test script was executed successfully.")

    print("")
    print("=" * 60)
    print("[UPDATED MEMORY SUMMARY]")
    print(memory.short_summary())

    print("")
    print("[MEMORY FILE]")
    print(memory.memory_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())