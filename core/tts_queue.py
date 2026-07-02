from __future__ import annotations

import queue
import threading
from typing import Callable, Optional


class TTSQueue:
    """
    Single-worker TTS queue.

    It prevents multiple XTTS calls from running at the same time while allowing
    the LLM streaming loop to continue generating the next sentence.
    """

    def __init__(
        self,
        xtts,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.xtts = xtts
        self.on_status = on_status

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name="LocalAI-TTSQueue",
            daemon=True,
        )
        self._thread.start()

    def put(self, text: str) -> None:
        text = str(text or "").strip()

        if not text:
            return

        if self._stop_event.is_set() or self._closed:
            return

        if not self._started:
            self.start()

        self._queue.put(text)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._queue.put(None)

    def stop(self) -> None:
        self._stop_event.set()
        self._closed = True

        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

        self._drain_pending_items()
        self._queue.put(None)

    def wait_until_done(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return

        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()

            try:
                if item is None:
                    break

                text = str(item or "").strip()

                if not text:
                    continue

                self._emit_status("🔊 Speaking...")
                self.xtts.speak(text)

            except Exception as exc:
                self._emit_status(f"TTS queue error: {exc}")

            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

        self._emit_status("TTS queue finished.")

    def _drain_pending_items(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
            except Exception:
                break

    def _emit_status(self, text: str) -> None:
        if self.on_status is None:
            return

        try:
            self.on_status(str(text))
        except Exception:
            pass
