from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.rag_store import RAGStore


def print_results(query: str) -> None:
    rag = RAGStore(root_dir=ROOT_DIR)

    print("")
    print("=" * 80)
    print(f"[QUERY] {query}")
    print("=" * 80)

    results = rag.search(
        query=query,
        top_k=3,
        min_score=0.01,
    )

    if not results:
        print("No results.")
        return

    for i, result in enumerate(results, start=1):
        print("")
        print("-" * 80)
        print(f"[RESULT {i}]")
        print(f"source: {result.source_path}")
        print(f"chunk: {result.chunk_index}")
        print(f"score: {result.score:.4f}")
        print("")
        print(result.text[:900])

    print("")
    print("=" * 80)
    print("[RAG CONTEXT]")
    print("=" * 80)
    print(rag.retrieve_context(query=query, top_k=3, max_chars=1200))


def main() -> int:
    test_queries = [
        "What is the LocalAI project architecture?",
        "What models does the project use?",
        "What is the current version of the project?",
        "What are the main bottlenecks?",
        "How does the voice assistant pipeline work?",
    ]

    for query in test_queries:
        print_results(query)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())