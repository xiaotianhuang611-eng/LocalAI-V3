from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RAGChunk:
    chunk_id: str
    source_path: str
    source_name: str
    chunk_index: int
    text: str


@dataclass
class RAGSearchResult:
    chunk_id: str
    source_path: str
    source_name: str
    chunk_index: int
    score: float
    text: str


class RAGStore:
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".py",
        ".json",
        ".csv",
        ".log",
        ".pdf",
        ".docx",
    }

    def __init__(
        self,
        root_dir: Path,
        knowledge_dir: Path | None = None,
        rag_dir: Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)

        self.knowledge_dir = (
            Path(knowledge_dir)
            if knowledge_dir is not None
            else self.root_dir / "data" / "knowledge"
        )

        self.rag_dir = (
            Path(rag_dir)
            if rag_dir is not None
            else self.root_dir / "data" / "rag"
        )

        self.index_path = self.rag_dir / "rag_index.pkl"
        self.meta_path = self.rag_dir / "rag_meta.json"

        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.rag_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: list[RAGChunk] = []
        self.vectorizer: Any | None = None
        self.matrix: Any | None = None
        self.is_loaded = False

    def create_bootstrap_project_profile(self, overwrite: bool = False) -> Path:
        path = self.knowledge_dir / "LocalAI_Project_Profile.md"

        if path.exists() and not overwrite:
            return path

        content = """
# LocalAI Project Profile

## Project Name
LocalAI_V3

## Project Path
C:\\Users\\111\\Desktop\\LocalAI_V3

## Project Direction
This project is an edge-native, privacy-preserving, multimodal speech assistant.

## Current Architecture
The system uses a PySide6 desktop UI and a ModelRuntime orchestration layer.

The current pipeline includes:
- VAD automatic voice activity detection
- faster-whisper ASR for speech recognition
- Gemma GGUF through llama.cpp for local language reasoning
- XTTS voice cloning for English spoken output
- Qwen2.5-VL GGUF for image understanding
- Local JSON memory for project and user preferences
- Personality modes
- Emotion engine
- English-only response mode for stable XTTS output

## Completed Versions
- V3.1: VAD auto-stop voice input
- V3.2: Personality Mode and Emotion Engine
- V3.3: Local Memory System
- V3.3.1: English-only voice output optimization
- V4.0: Local RAG Knowledge Base

## Technical Stack
- UI: PySide6
- ASR: faster-whisper small.en
- LLM: Gemma GGUF
- Runtime: llama-cpp-python
- TTS: Coqui XTTS v2
- Vision: Qwen2.5-VL GGUF with mmproj
- Memory: local JSON structured memory
- RAG: local document retrieval

## Engineering Bottlenecks
The main bottlenecks are XTTS synthesis time, Gemma prompt length, GPU VRAM limits, Qwen-VL loading and unloading, and RAG context length.

## Recommended Fast Mode
For daily fast voice chat:
- Persona OFF
- Emotion OFF
- Memory OFF
- RAG OFF
- English-only output
- Direct XTTS playback

## Recommended Knowledge Mode
For project, thesis, coursework, or document questions:
- Persona ON
- Memory ON
- RAG ON
- Use Project Engineer or Thesis Assistant mode
""".strip()

        path.write_text(content, encoding="utf-8")
        return path

    def build_index(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        max_features: int = 50000,
    ) -> dict[str, Any]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:
            raise RuntimeError(
                "scikit-learn is required. Install it with: "
                "python -m pip install scikit-learn"
            ) from exc

        start = time.perf_counter()

        documents = self._collect_documents()

        if not documents:
            raise RuntimeError(f"No supported documents found in: {self.knowledge_dir}")

        chunks: list[RAGChunk] = []

        for file_path, text in documents:
            file_chunks = self._chunk_text(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            rel_path = self._relative_path(file_path)

            for index, chunk_text in enumerate(file_chunks):
                chunk_id = f"{rel_path}::chunk_{index}"

                chunks.append(
                    RAGChunk(
                        chunk_id=chunk_id,
                        source_path=rel_path,
                        source_name=file_path.name,
                        chunk_index=index,
                        text=chunk_text,
                    )
                )

        if not chunks:
            raise RuntimeError("Documents were found, but no valid chunks were created.")

        texts = [chunk.text for chunk in chunks]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            strip_accents="unicode",
        )

        matrix = vectorizer.fit_transform(texts)

        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.is_loaded = True

        self._save_index()

        elapsed = time.perf_counter() - start

        meta = {
            "version": "V4.0 Lite RAG",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "knowledge_dir": str(self.knowledge_dir),
            "index_path": str(self.index_path),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_features": max_features,
            "elapsed_seconds": round(elapsed, 3),
        }

        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return meta

    def load_index(self) -> None:
        if self.is_loaded:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"RAG index not found: {self.index_path}. "
                "Run tools/build_rag_index.py first."
            )

        with self.index_path.open("rb") as f:
            payload = pickle.load(f)

        self.chunks = [RAGChunk(**item) for item in payload.get("chunks", [])]
        self.vectorizer = payload.get("vectorizer")
        self.matrix = payload.get("matrix")
        self.is_loaded = True

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.02,
    ) -> list[RAGSearchResult]:
        query = str(query or "").strip()

        if not query:
            return []

        if not self.is_loaded:
            self.load_index()

        if self.vectorizer is None or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()

        ranked_indices = scores.argsort()[::-1]

        results: list[RAGSearchResult] = []

        for idx in ranked_indices[: max(top_k * 3, top_k)]:
            score = float(scores[idx])

            if score < min_score:
                continue

            chunk = self.chunks[int(idx)]

            results.append(
                RAGSearchResult(
                    chunk_id=chunk.chunk_id,
                    source_path=chunk.source_path,
                    source_name=chunk.source_name,
                    chunk_index=chunk.chunk_index,
                    score=score,
                    text=chunk.text,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 1200,
        min_score: float = 0.02,
    ) -> str:
        results = self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        if not results:
            return ""

        parts: list[str] = []
        parts.append("Local RAG knowledge context:")

        for i, result in enumerate(results, start=1):
            parts.append("")
            parts.append(
                f"[RAG-{i}] Source: {result.source_path} | "
                f"chunk={result.chunk_index} | score={result.score:.4f}"
            )
            parts.append(result.text.strip())

        text = "\n".join(parts).strip()

        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0].strip()

        return text

    def status(self) -> dict[str, Any]:
        status = {
            "version": "V4.0 Lite RAG",
            "knowledge_dir": str(self.knowledge_dir),
            "rag_dir": str(self.rag_dir),
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
            "is_loaded": self.is_loaded,
            "chunk_count": len(self.chunks),
        }

        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                status.update(meta)
            except Exception:
                pass

        return status

    def _save_index(self) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
        }

        with self.index_path.open("wb") as f:
            pickle.dump(payload, f)

    def _collect_documents(self) -> list[tuple[Path, str]]:
        documents: list[tuple[Path, str]] = []

        for file_path in sorted(self.knowledge_dir.rglob("*")):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                text = self._read_file(file_path)
                text = self._clean_text(text)

                if len(text) >= 30:
                    documents.append((file_path, text))

            except Exception as exc:
                print(f"[RAG] Skipped file: {file_path} | reason: {exc}")

        return documents

    def _read_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(file_path)

        if suffix == ".docx":
            return self._read_docx(file_path)

        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("Install pypdf first: python -m pip install pypdf") from exc

        reader = PdfReader(str(file_path))
        pages: list[str] = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")

        return "\n".join(pages)

    def _read_docx(self, file_path: Path) -> str:
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError(
                "Install python-docx first: python -m pip install python-docx"
            ) from exc

        doc = Document(str(file_path))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        text = self._clean_text(text)

        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            if end < len(text):
                cut_positions = [
                    chunk.rfind(". "),
                    chunk.rfind("? "),
                    chunk.rfind("! "),
                    chunk.rfind("\n"),
                    chunk.rfind("。"),
                    chunk.rfind("？"),
                    chunk.rfind("！"),
                ]

                best_cut = max(cut_positions)

                if best_cut > int(chunk_size * 0.55):
                    chunk = chunk[: best_cut + 1]
                    end = start + best_cut + 1

            chunk = chunk.strip()

            if len(chunk) >= 30:
                chunks.append(chunk)

            next_start = max(0, end - chunk_overlap)

            if next_start <= start:
                break

            start = next_start

        return chunks

    def _clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        return text

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_dir))
        except Exception:
            return str(path)