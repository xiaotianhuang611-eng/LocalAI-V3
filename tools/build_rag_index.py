from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.rag_store import RAGStore


def main() -> int:
    rag = RAGStore(root_dir=ROOT_DIR)

    profile_path = rag.create_bootstrap_project_profile(overwrite=False)

    print("=" * 70)
    print("[RAG] Bootstrap project profile")
    print(profile_path)

    print("")
    print("=" * 70)
    print("[RAG] Building index...")

    meta = rag.build_index(
        chunk_size=900,
        chunk_overlap=150,
        max_features=50000,
    )

    print("")
    print("=" * 70)
    print("[RAG] Build complete")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    print("")
    print("=" * 70)
    print("[RAG] Status")
    print(json.dumps(rag.status(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())