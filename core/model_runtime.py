from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Callable

from core.asr import FasterWhisperASR
from core.gemma_chat import GemmaChat
from core.qwen_vision import QwenVision
from core.sentence_buffer import SentenceBuffer
from core.tts_queue import TTSQueue
from core.vad_recorder import EnergyVADRecorder
from core.xtts_fast import XTTSFast


class ModelRuntime:
    def __init__(
        self,
        root_dir: Path,
        personality_mode: str = "project_engineer",
        personality_enabled: bool = True,
        emotion_enabled: bool = True,
        memory_enabled: bool = True,
        rag_enabled: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)

        self.gemma = GemmaChat(
            root_dir=self.root_dir,
            personality_mode=personality_mode,
            personality_enabled=personality_enabled,
            emotion_enabled=emotion_enabled,
            memory_enabled=memory_enabled,
            rag_enabled=rag_enabled,
        )

        self.xtts = XTTSFast(root_dir=self.root_dir)
        self.asr = FasterWhisperASR(root_dir=self.root_dir, model_name="small.en")

        self._active_tts_queue: TTSQueue | None = None

    def load(self) -> None:
        print("[Runtime] Loading Gemma...")
        self.gemma.load()

        print("[Runtime] Loading XTTSFast...")
        self.xtts.load()

        print("[Runtime] Ready.")

    def load_asr(self) -> None:
        print("[Runtime] Loading ASR...")
        self.asr.load()

    def set_personality_mode(self, mode: str) -> None:
        self.gemma.set_personality_mode(mode)

    def get_personality_mode(self) -> str:
        return self.gemma.get_personality_mode()

    def set_personality_enabled(self, enabled: bool) -> None:
        self.gemma.set_personality_enabled(enabled)

    def get_personality_enabled(self) -> bool:
        return self.gemma.get_personality_enabled()

    def set_emotion_enabled(self, enabled: bool) -> None:
        self.gemma.set_emotion_enabled(enabled)

    def get_emotion_summary(self) -> str:
        return self.gemma.get_emotion_summary()

    def set_memory_enabled(self, enabled: bool) -> None:
        self.gemma.set_memory_enabled(enabled)

    def get_memory_summary(self) -> str:
        return self.gemma.get_memory_summary()

    def get_memory_context(self, query: str = "") -> str:
        return self.gemma.get_memory_context(query=query)

    def add_memory_fact(self, fact: str) -> None:
        self.gemma.add_memory_fact(fact)

    def clear_recent_memory_events(self) -> None:
        self.gemma.clear_recent_memory_events()

    def set_rag_enabled(self, enabled: bool) -> None:
        self.gemma.set_rag_enabled(enabled)

    def get_rag_enabled(self) -> bool:
        return self.gemma.get_rag_enabled()

    def get_rag_summary(self) -> str:
        return self.gemma.get_rag_summary()

    def listen(self, seconds: int = 5) -> str:
        if self.asr.model is None:
            self.load_asr()

        return self.asr.record_and_transcribe(seconds=seconds)

    def listen_auto(self, max_seconds: int = 30) -> str:
        if self.asr.model is None:
            self.load_asr()

        recorder = EnergyVADRecorder(
            root_dir=self.root_dir,
            start_threshold=0.012,
            end_threshold=0.008,
            silence_end_seconds=0.85,
            pre_roll_seconds=0.35,
            min_record_seconds=0.85,
            max_record_seconds=float(max_seconds),
        )

        wav_path = recorder.record_auto_stop()

        return self.asr.transcribe_file(wav_path)

    def ask(self, text: str) -> str:
        return self.gemma.ask(text)

    def speak(self, text: str) -> None:
        """
        Fast direct XTTS output.

        Gemma is forced to answer in English, so no second rewrite step is used.
        """
        self.xtts.speak(text)

    def ask_stream_and_speak(
        self,
        text: str,
        live_mode: bool = True,
        live_fast_mode: bool = True,
        max_tokens: int = 100,
        sentence_max_chars: int = 220,
        on_token: Callable[[str], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """
        Streaming live voice path.

        Gemma streams text tokens. SentenceBuffer groups tokens into short
        sentences. TTSQueue speaks each sentence in order while Gemma continues
        generating the rest of the response.
        """
        text = str(text or "").strip()

        if not text:
            return "I did not hear anything clearly."

        def emit_status(status: str) -> None:
            if on_status is None:
                return

            try:
                on_status(str(status))
            except Exception:
                pass

        def emit_token(token: str) -> None:
            if on_token is None:
                return

            try:
                on_token(str(token))
            except Exception:
                pass

        def emit_sentence(sentence: str) -> None:
            if on_sentence is None:
                return

            try:
                on_sentence(str(sentence))
            except Exception:
                pass

        def stop_requested() -> bool:
            if should_stop is None:
                return False

            try:
                return bool(should_stop())
            except Exception:
                return False

        sentence_buffer = SentenceBuffer(
            max_chars=int(sentence_max_chars),
            min_chars=8,
        )

        tts_queue = TTSQueue(
            xtts=self.xtts,
            on_status=emit_status,
        )

        self._active_tts_queue = tts_queue
        tts_queue.start()

        raw_parts: list[str] = []

        emit_status("🧠 Thinking and streaming...")

        try:
            for token_text in self.gemma.ask_stream(
                text=text,
                live_mode=live_mode,
                live_fast_mode=live_fast_mode,
                max_tokens=int(max_tokens),
                temperature=0.40,
                top_p=0.90,
            ):
                if stop_requested():
                    emit_status("Streaming stopped.")
                    break

                token_text = str(token_text or "")

                if not token_text:
                    continue

                raw_parts.append(token_text)
                emit_token(token_text)

                ready_sentences = sentence_buffer.add(token_text)

                for sentence in ready_sentences:
                    if stop_requested():
                        break

                    safe_sentence = self._make_stream_sentence_tts_safe(sentence)

                    if not safe_sentence:
                        continue

                    emit_sentence(safe_sentence)
                    emit_status("🔊 Speaking sentence...")
                    tts_queue.put(safe_sentence)

                if stop_requested():
                    emit_status("Streaming stopped.")
                    break

            if not stop_requested():
                for sentence in sentence_buffer.flush():
                    safe_sentence = self._make_stream_sentence_tts_safe(sentence)

                    if not safe_sentence:
                        continue

                    emit_sentence(safe_sentence)
                    emit_status("🔊 Speaking final sentence...")
                    tts_queue.put(safe_sentence)

        finally:
            if stop_requested():
                tts_queue.stop()
            else:
                tts_queue.close()

            tts_queue.wait_until_done()
            self._active_tts_queue = None

        raw_response = "".join(raw_parts).strip()
        final_response = self.gemma.clean_response_for_display(raw_response)

        emit_status("✅ Ready.")

        return final_response

    def stop_streaming_speech(self) -> None:
        try:
            if self._active_tts_queue is not None:
                self._active_tts_queue.stop()
        except Exception:
            pass

        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def ask_and_speak(self, text: str) -> str:
        response = self.ask(text)
        self.speak(response)
        return response

    def listen_ask_and_speak(self, seconds: int = 5) -> tuple[str, str]:
        user_text = self.listen(seconds=seconds)

        if not user_text:
            user_text = "I did not hear anything clearly."

        response = self.ask(user_text)
        self.speak(response)

        return user_text, response

    def listen_auto_ask_and_speak(self, max_seconds: int = 30) -> tuple[str, str]:
        user_text = self.listen_auto(max_seconds=max_seconds)

        if not user_text:
            user_text = "I did not hear anything clearly."

        response = self.ask(user_text)
        self.speak(response)

        return user_text, response

    def analyze_image_safe(self, image_path: Path, question: str = "What is in this image?") -> str:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("[Runtime] Preparing memory for Qwen-VL...")
        self.unload_for_vision()

        qwen = QwenVision(root_dir=self.root_dir)

        safe_question = (
            str(question or "What is in this image?").strip()
            + "\n\nPlease answer in English only. Do not output Chinese characters."
        )

        try:
            print("[Runtime] Loading Qwen-VL...")
            qwen.load()

            vision_answer = qwen.describe_image(
                image_path=image_path,
                question=safe_question,
            )

        finally:
            qwen.unload()
            del qwen
            self._release_memory()

        print("[Runtime] Reloading Gemma + XTTSFast after Qwen-VL...")
        self.load()

        return vision_answer

    def analyze_image_safe_and_speak(
        self,
        image_path: Path,
        question: str = "What is in this image?",
    ) -> tuple[str, str]:
        vision_answer = self.analyze_image_safe(
            image_path=image_path,
            question=question,
        )

        spoken_answer = self._make_vision_answer_tts_safe(vision_answer)

        print("[Runtime] Speaking vision answer directly in English...")
        self.speak(spoken_answer)

        return vision_answer, spoken_answer

    def _make_stream_sentence_tts_safe(self, text: str) -> str:
        text = str(text or "").strip()

        if not text:
            return ""

        prefixes = [
            "Assistant:",
            "assistant:",
            "AI:",
            "Mochi:",
            "Spoken answer:",
            "English spoken answer:",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    changed = True

        cleaned_chars: list[str] = []

        for ch in text:
            code = ord(ch)

            if 0x4E00 <= code <= 0x9FFF:
                continue

            cleaned_chars.append(ch)

        text = "".join(cleaned_chars).strip()
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        if not text:
            return ""

        max_chars = 260

        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].strip()
            text = text.rstrip(".,;:") + "."

        return text

    def _make_vision_answer_tts_safe(self, text: str) -> str:
        """
        Lightweight safety cleanup only.

        This does not call Gemma again.
        It only prevents XTTS from reading extremely long image answers.
        """
        text = str(text or "").strip()

        if not text:
            return "I could not find enough visual information to describe."

        cleaned_chars: list[str] = []

        for ch in text:
            code = ord(ch)

            if 0x4E00 <= code <= 0x9FFF:
                continue

            cleaned_chars.append(ch)

        text = "".join(cleaned_chars).strip()
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        if not text:
            return "I could not find enough visual information to describe."

        max_chars = 450

        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].strip()
            text = text.rstrip(".,;:") + "."

        return text

    def unload_for_vision(self) -> None:
        print("[Runtime] Unloading Gemma / XTTSFast / ASR for vision...")

        try:
            self.stop_streaming_speech()
        except Exception:
            pass

        try:
            self.gemma.llm = None
        except Exception:
            pass

        try:
            self.xtts.tts = None
        except Exception:
            pass

        try:
            self.asr.model = None
        except Exception:
            pass

        self._release_memory()

        print("[Runtime] Memory prepared for vision.")

    def reset_conversation(self) -> None:
        self.gemma.reset_history()

    def get_conversation_text(self) -> str:
        return self.gemma.get_history_text()

    def _release_memory(self) -> None:
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except Exception:
            pass

        time.sleep(1.0)
