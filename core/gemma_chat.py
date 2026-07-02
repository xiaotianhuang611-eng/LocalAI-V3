from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from llama_cpp import Llama

from core.emotion_engine import EmotionEngine
from core.memory_store import MemoryStore
from core.persona import DEFAULT_PERSONA_MODE, get_persona_mode, normalize_persona_mode
from core.prompt_builder import build_gemma_prompt
from core.rag_store import RAGStore


class GemmaChat:
    def __init__(
        self,
        root_dir: Path,
        model_name: str = "google_gemma-4-E4B-it-Q5_K_M.gguf",
        max_history_turns: int = 6,
        personality_mode: str = DEFAULT_PERSONA_MODE,
        personality_enabled: bool = True,
        emotion_enabled: bool = True,
        memory_enabled: bool = True,
        rag_enabled: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_path = self.root_dir / "models" / model_name
        self.max_history_turns = int(max_history_turns)

        self.personality_mode = normalize_persona_mode(personality_mode)
        self.personality_enabled = bool(personality_enabled)
        self.emotion_enabled = bool(emotion_enabled)
        self.memory_enabled = bool(memory_enabled)
        self.rag_enabled = bool(rag_enabled)

        self.emotion = EmotionEngine()
        self.memory = MemoryStore(root_dir=self.root_dir)
        self.rag = RAGStore(root_dir=self.root_dir)

        self.llm: Llama | None = None
        self.history: list[tuple[str, str]] = []
        self.last_response: str = ""

    def load(self) -> None:
        if self.llm is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Gemma model not found: {self.model_path}")

        print(f"[Gemma] Loading model: {self.model_path}")

        start = time.perf_counter()

        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=256,
            flash_attn=True,
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        print(f"[Gemma] Loaded in {elapsed:.2f}s")
        print(f"[Gemma] Personality mode: {self.personality_mode}")
        print(f"[Gemma] Personality enabled: {self.personality_enabled}")
        print(f"[Gemma] Emotion enabled: {self.emotion_enabled}")
        print(f"[Gemma] Memory enabled: {self.memory_enabled}")
        print(f"[Gemma] RAG enabled: {self.rag_enabled}")
        print(f"[Gemma] Memory: {self.memory.short_summary()}")
        print(f"[Gemma] RAG: {self.get_rag_summary()}")

    def set_personality_mode(self, mode: str) -> None:
        self.personality_mode = normalize_persona_mode(mode)
        persona = get_persona_mode(self.personality_mode)
        print(f"[Gemma] Personality mode set to: {persona.display_name}")

    def get_personality_mode(self) -> str:
        return self.personality_mode

    def set_personality_enabled(self, enabled: bool) -> None:
        self.personality_enabled = bool(enabled)
        print(f"[Gemma] Personality enabled: {self.personality_enabled}")

    def get_personality_enabled(self) -> bool:
        return self.personality_enabled

    def set_emotion_enabled(self, enabled: bool) -> None:
        self.emotion_enabled = bool(enabled)
        print(f"[Gemma] Emotion enabled: {self.emotion_enabled}")

    def get_emotion_summary(self) -> str:
        return self.emotion.summary_text()

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_enabled = bool(enabled)
        print(f"[Gemma] Memory enabled: {self.memory_enabled}")

    def get_memory_summary(self) -> str:
        return self.memory.short_summary()

    def get_memory_context(self, query: str = "") -> str:
        return self.memory.retrieve_context(query=query)

    def add_memory_fact(self, fact: str) -> None:
        self.memory.add_stable_fact(fact)

    def clear_recent_memory_events(self) -> None:
        self.memory.clear_recent_events()
        print("[Gemma] Recent memory events cleared.")

    def set_rag_enabled(self, enabled: bool) -> None:
        self.rag_enabled = bool(enabled)
        print(f"[Gemma] RAG enabled: {self.rag_enabled}")

    def get_rag_enabled(self) -> bool:
        return self.rag_enabled

    def get_rag_summary(self) -> str:
        try:
            status = self.rag.status()
            return (
                f"RAG | enabled={self.rag_enabled} | "
                f"index_exists={status.get('index_exists')} | "
                f"chunks={status.get('chunk_count', 0)}"
            )
        except Exception as exc:
            return f"RAG | enabled={self.rag_enabled} | error={exc}"

    def ask(self, text: str) -> str:
        if self.llm is None:
            self.load()

        text = str(text).strip()

        if not text:
            return "I did not receive any input."

        prompt = self._build_prompt_for_request(
            text=text,
            live_mode=False,
            live_fast_mode=False,
        )

        print(f"[Gemma] Asking with mode={self.personality_mode}")
        print(f"[Gemma] Personality enabled={self.personality_enabled}")
        print(f"[Gemma] Emotion enabled={self.emotion_enabled}")
        print(f"[Gemma] Memory enabled={self.memory_enabled}")
        print(f"[Gemma] RAG enabled={self.rag_enabled}")

        start = time.perf_counter()

        output = self.llm(
            prompt,
            max_tokens=160,
            temperature=0.45,
            top_p=0.9,
            stop=["User:", "\nUser:", "<eos>", "</s>"],
        )

        elapsed = time.perf_counter() - start

        response = self._extract_text(output)
        response = self.clean_response_for_display(response)

        self._finalize_response(
            user_text=text,
            response=response,
        )

        print(f"[Gemma] Response in {elapsed:.2f}s")
        print(f"[Gemma] Emotion: {self.get_emotion_summary()}")
        print(f"[Gemma] Memory: {self.get_memory_summary()}")
        print(f"[Gemma] RAG: {self.get_rag_summary()}")

        return response

    def ask_stream(
        self,
        text: str,
        live_mode: bool = True,
        live_fast_mode: bool = True,
        max_tokens: int = 100,
        temperature: float = 0.40,
        top_p: float = 0.90,
    ) -> Generator[str, None, None]:
        """
        Stream Gemma output token by token.

        The final cleaned response is stored in self.last_response and added to
        history/memory after generation finishes normally.
        """
        if self.llm is None:
            self.load()

        text = str(text).strip()

        if not text:
            self.last_response = "I did not receive any input."
            yield self.last_response
            return

        prompt = self._build_prompt_for_request(
            text=text,
            live_mode=live_mode,
            live_fast_mode=live_fast_mode,
        )

        print(f"[Gemma] Streaming with mode={self.personality_mode}")
        print(f"[Gemma] Live mode={live_mode}")
        print(f"[Gemma] Live fast mode={live_fast_mode}")
        print(f"[Gemma] Personality enabled={self.personality_enabled}")
        print(f"[Gemma] Emotion enabled={self.emotion_enabled}")
        print(f"[Gemma] Memory enabled={self.memory_enabled}")
        print(f"[Gemma] RAG enabled={self.rag_enabled}")

        start = time.perf_counter()
        raw_parts: list[str] = []

        stream = self.llm(
            prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop=["User:", "\nUser:", "<eos>", "</s>"],
            stream=True,
        )

        for chunk in stream:
            token_text = self._extract_stream_text(chunk)

            if not token_text:
                continue

            raw_parts.append(token_text)
            yield token_text

        elapsed = time.perf_counter() - start

        response = "".join(raw_parts)
        response = self.clean_response_for_display(response)

        self.last_response = response

        self._finalize_response(
            user_text=text,
            response=response,
        )

        print(f"[Gemma] Stream response in {elapsed:.2f}s")
        print(f"[Gemma] Emotion: {self.get_emotion_summary()}")
        print(f"[Gemma] Memory: {self.get_memory_summary()}")
        print(f"[Gemma] RAG: {self.get_rag_summary()}")

    def rewrite_for_voice(self, content: str) -> str:
        """
        Optional fallback only.

        The current English-only fast path does not normally call this.
        """
        if self.llm is None:
            self.load()

        content = str(content).strip()

        if not content:
            return "I do not have enough information to answer."

        prompt = f"""
System:
Rewrite the following content into short natural English for voice output.

Rules:
- English only.
- No Chinese characters.
- No markdown.
- One or two short spoken sentences.
- Preserve the core meaning.

Content:
{content}

English spoken answer:
""".strip()

        output = self.llm(
            prompt,
            max_tokens=100,
            temperature=0.30,
            top_p=0.90,
            stop=["User:", "\nUser:", "<eos>", "</s>"],
        )

        response = self._extract_text(output)
        response = self.clean_response_for_display(response)

        return response

    def reset_history(self) -> None:
        self.history.clear()
        self.emotion.reset()
        print("[Gemma] Conversation history and emotion state cleared.")

    def get_history_text(self) -> str:
        if not self.history:
            return ""

        lines: list[str] = []

        for user_text, assistant_text in self.history[-self.max_history_turns:]:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")

        return "\n".join(lines)

    def clean_response_for_display(self, text: str) -> str:
        text = self._clean_response(text)
        text = self._force_english_safe_response(text)
        return text

    def _build_prompt_for_request(
        self,
        text: str,
        live_mode: bool = False,
        live_fast_mode: bool = False,
    ) -> str:
        text = str(text).strip()

        if self.emotion_enabled and self.personality_enabled and not live_fast_mode:
            self.emotion.update_from_user_text(text)
            emotion_prompt = self.emotion.style_prompt()
        elif self.emotion_enabled and self.personality_enabled:
            # Keep the emotion state lightly updated, but do not inject the
            # emotion prompt into the low-latency live prompt.
            self.emotion.update_from_user_text(text)
            emotion_prompt = ""
        else:
            emotion_prompt = ""

        if self.memory_enabled and not live_fast_mode:
            memory_context = self.memory.retrieve_context(query=text)
        else:
            memory_context = ""

        rag_context = ""

        if self.rag_enabled and not live_fast_mode:
            rag_context = self._retrieve_rag_context(
                query=text,
                top_k=3,
                max_chars=1200,
                min_score=0.02,
            )
        elif self.rag_enabled and live_fast_mode:
            print("[Gemma] Live fast mode is active; RAG context skipped for low latency.")

        prompt = build_gemma_prompt(
            user_text=text,
            history_text=self.get_history_text(),
            personality_mode=self.personality_mode,
            personality_enabled=self.personality_enabled,
            emotion_prompt=emotion_prompt,
            memory_context=memory_context,
            rag_context=rag_context,
            live_mode=live_mode,
        )

        return prompt

    def _retrieve_rag_context(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 1200,
        min_score: float = 0.02,
    ) -> str:
        try:
            rag_context = self.rag.retrieve_context(
                query=query,
                top_k=top_k,
                max_chars=max_chars,
                min_score=min_score,
            )

            if rag_context:
                print("[Gemma] RAG context retrieved.")
            else:
                print("[Gemma] RAG enabled, but no relevant context found.")

            return rag_context

        except FileNotFoundError:
            print("[Gemma] RAG index not found. Run tools/build_rag_index.py first.")
            return ""

        except Exception as exc:
            print(f"[Gemma] RAG retrieval failed: {exc}")
            return ""

    def _finalize_response(self, user_text: str, response: str) -> None:
        response = self.clean_response_for_display(response)
        self.last_response = response

        self._add_turn(user_text, response)

        if self.memory_enabled:
            self.memory.update_from_interaction(
                user_text=user_text,
                assistant_text=response,
                personality_mode=self.personality_mode,
                emotion_summary=self.get_emotion_summary(),
            )

    def _add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append((str(user_text), str(assistant_text)))

        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def _extract_text(self, output) -> str:
        try:
            return str(output["choices"][0]["text"]).strip()
        except Exception:
            return str(output).strip()

    def _extract_stream_text(self, chunk) -> str:
        try:
            return str(chunk["choices"][0].get("text", ""))
        except Exception:
            return ""

    def _clean_response(self, text: str) -> str:
        text = str(text or "").strip()

        prefixes = [
            "Assistant:",
            "assistant:",
            "Mochi:",
            "AI:",
            "English spoken answer:",
            "Spoken answer:",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    changed = True

        return text.strip()

    def _force_english_safe_response(self, text: str) -> str:
        """
        Safety layer for XTTS.

        The prompt already asks for English-only output.
        This only removes accidental Chinese characters if the model leaks them.
        """
        text = str(text or "").strip()

        if not text:
            return "I am ready. Please ask again."

        cleaned_chars: list[str] = []

        for ch in text:
            code = ord(ch)

            if 0x4E00 <= code <= 0x9FFF:
                continue

            cleaned_chars.append(ch)

        cleaned = "".join(cleaned_chars).strip()
        cleaned = " ".join(cleaned.split())

        if not cleaned:
            return "I am ready. Please ask again."

        english_letters = sum(1 for ch in cleaned if ch.isascii() and ch.isalpha())

        if english_letters < 4:
            return "I am ready. Please ask again."

        return cleaned
