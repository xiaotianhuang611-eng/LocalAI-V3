from __future__ import annotations

import re


class SentenceBuffer:
    """
    Convert streamed LLM tokens into sentence-sized chunks for TTS.

    This is not token-level audio streaming. It is sentence-level streaming:
    as soon as a complete sentence is detected, the sentence is returned so
    the TTS queue can start speaking before the full LLM response is finished.
    """

    def __init__(
        self,
        max_chars: int = 220,
        min_chars: int = 8,
    ) -> None:
        self.max_chars = int(max_chars)
        self.min_chars = int(min_chars)
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        text = str(text or "")

        if not text:
            return []

        self._buffer += text
        self._buffer = self._normalise_spacing(self._buffer)

        ready: list[str] = []

        while True:
            sentence, remaining = self._extract_one_sentence(self._buffer)

            if sentence is None:
                break

            self._buffer = remaining
            sentence = sentence.strip()

            if sentence:
                ready.append(sentence)

        if len(self._buffer) >= self.max_chars:
            forced, remaining = self._split_long_buffer(self._buffer)
            self._buffer = remaining

            if forced.strip():
                ready.append(forced.strip())

        return ready

    def flush(self) -> list[str]:
        remaining = self._buffer.strip()
        self._buffer = ""

        if not remaining:
            return []

        return [remaining]

    def reset(self) -> None:
        self._buffer = ""

    def _extract_one_sentence(self, text: str) -> tuple[str | None, str]:
        text = str(text or "")

        if len(text.strip()) < self.min_chars:
            return None, text

        # Prefer natural English sentence boundaries.
        match = re.search(r"([.!?])(\s+|$)", text)

        if match is None:
            return None, text

        end_index = match.end(1)
        sentence = text[:end_index]
        remaining = text[end_index:]

        if len(sentence.strip()) < self.min_chars:
            return None, text

        return sentence, remaining

    def _split_long_buffer(self, text: str) -> tuple[str, str]:
        text = str(text or "")

        if len(text) <= self.max_chars:
            return text, ""

        split_at = text.rfind(" ", 0, self.max_chars)

        if split_at < max(40, self.max_chars // 3):
            split_at = self.max_chars

        return text[:split_at], text[split_at:]

    def _normalise_spacing(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text
