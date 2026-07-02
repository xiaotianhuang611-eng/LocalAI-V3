from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.web_knowledge import WebKnowledgeIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search or crawl public web pages and save them into LocalAI RAG knowledge folder.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Web search query to ingest into RAG.")
    group.add_argument("--url", type=str, help="Direct URL to crawl into RAG.")

    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum number of search results to crawl. Default: 3.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the RAG index after saving crawled documents.",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Disable robots.txt checking. Use only for pages you are allowed to crawl.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ingestor = WebKnowledgeIngestor(
        root_dir=ROOT_DIR,
        respect_robots=not bool(args.ignore_robots),
    )

    try:
        if args.url:
            print(f"[WebKnowledge] Crawling URL: {args.url}")
            document = ingestor.crawl_and_save(args.url)
            print("[WebKnowledge] Saved document:")
            print(f"  Title: {document.title}")
            print(f"  URL:   {document.url}")
            print(f"  Path:  {document.saved_path}")
            print(f"  Chars: {document.text_chars}")

        else:
            print(f"[WebKnowledge] Searching web: {args.query}")
            documents, errors = ingestor.search_and_ingest(
                query=args.query,
                max_results=args.max_results,
            )

            print(f"[WebKnowledge] Saved {len(documents)} document(s):")

            for index, document in enumerate(documents, start=1):
                print(f"  {index}. {document.title}")
                print(f"     URL:  {document.url}")
                print(f"     Path: {document.saved_path}")
                print(f"     Chars: {document.text_chars}")

            if errors:
                print("[WebKnowledge] Some pages failed:")

                for error in errors:
                    print(f"  - {error}")

        if args.rebuild:
            print("[WebKnowledge] Rebuilding RAG index...")
            output = ingestor.rebuild_rag_index()
            print(output)

        print("[WebKnowledge] Done.")
        return 0

    except Exception as exc:
        print(f"[WebKnowledge] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
