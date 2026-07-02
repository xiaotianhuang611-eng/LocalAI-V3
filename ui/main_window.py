from __future__ import annotations

import sys
import time
from pathlib import Path

import sounddevice as sd

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPainter, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import AppSettings
from core.model_runtime import ModelRuntime
from core.persona import get_persona_options


class BackgroundCanvas(QWidget):
    def __init__(self, bg_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.bg_path = Path(bg_path)
        self.pixmap = QPixmap(str(self.bg_path))
        self.setFixedSize(1200, 980)
        self.setObjectName("backgroundCanvas")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if not self.pixmap.isNull():
            painter.drawPixmap(self.rect(), self.pixmap)
        else:
            painter.fillRect(self.rect(), Qt.white)


class LoadRuntimeWorker(QThread):
    progress = Signal(str)
    loaded = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        personality_mode: str = "project_engineer",
        personality_enabled: bool = True,
        emotion_enabled: bool = True,
        memory_enabled: bool = True,
        rag_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.personality_mode = personality_mode
        self.personality_enabled = bool(personality_enabled)
        self.emotion_enabled = bool(emotion_enabled)
        self.memory_enabled = bool(memory_enabled)
        self.rag_enabled = bool(rag_enabled)

    def run(self) -> None:
        try:
            self.progress.emit("Loading Gemma and XTTSFast...")

            runtime = ModelRuntime(
                root_dir=ROOT_DIR,
                personality_mode=self.personality_mode,
                personality_enabled=self.personality_enabled,
                emotion_enabled=self.emotion_enabled,
                memory_enabled=self.memory_enabled,
                rag_enabled=self.rag_enabled,
            )

            runtime.load()

            self.progress.emit("Loading ASR...")
            runtime.load_asr()

            self.progress.emit("Runtime ready.")
            self.loaded.emit(runtime)

        except Exception as exc:
            self.error.emit(str(exc))


class AskSpeakWorker(QThread):
    progress = Signal(str)
    ai_response = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(self, runtime: ModelRuntime, text: str) -> None:
        super().__init__()
        self.runtime = runtime
        self.text = text

    def run(self) -> None:
        try:
            self.progress.emit("Gemma is thinking...")

            response = self.runtime.ask(self.text)
            self.ai_response.emit(response)

            self.progress.emit("XTTSFast is speaking...")
            self.runtime.speak(response)

            self.progress.emit("Done.")
            self.completed.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class VoiceAssistantWorker(QThread):
    progress = Signal(str)
    user_text = Signal(str)
    ai_response = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(self, runtime: ModelRuntime, seconds: int = 30) -> None:
        super().__init__()
        self.runtime = runtime
        self.seconds = int(seconds)

    def run(self) -> None:
        try:
            self.progress.emit(f"Auto listening, max {self.seconds}s...")

            text = self.runtime.listen_auto(max_seconds=self.seconds)

            if not text:
                text = "I did not hear anything clearly."

            self.user_text.emit(text)

            self.progress.emit("Gemma is thinking...")

            response = self.runtime.ask(text)
            self.ai_response.emit(response)

            self.progress.emit("XTTSFast is speaking...")
            self.runtime.speak(response)

            self.progress.emit("Done.")
            self.completed.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class LiveChatWorker(QThread):
    progress = Signal(str)
    user_text = Signal(str)
    partial_response = Signal(str)
    speech_sentence = Signal(str)
    ai_response = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(
        self,
        runtime: ModelRuntime,
        seconds: int = 20,
        pause_after_speech: float = 0.15,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.seconds = int(seconds)
        self.pause_after_speech = float(pause_after_speech)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

        try:
            self.runtime.stop_streaming_speech()
        except Exception:
            pass

        try:
            sd.stop()
        except Exception:
            pass

    def _should_stop(self) -> bool:
        return bool(self._stop_requested)

    def run(self) -> None:
        turn_index = 0

        try:
            self.progress.emit("Live chat started.")

            while not self._stop_requested:
                turn_index += 1

                self.progress.emit("🟢 Listening... Please speak now.")

                text = self.runtime.listen_auto(max_seconds=self.seconds)
                text = str(text or "").strip()

                if self._stop_requested:
                    break

                if not text:
                    self.progress.emit("No clear speech detected. Listening again...")
                    time.sleep(0.15)
                    continue

                self.user_text.emit(text)

                self.progress.emit("🧠 Streaming response...")

                response = self.runtime.ask_stream_and_speak(
                    text=text,
                    live_mode=True,
                    live_fast_mode=True,
                    max_tokens=100,
                    sentence_max_chars=220,
                    on_token=lambda token: self.partial_response.emit(token),
                    on_sentence=lambda sentence: self.speech_sentence.emit(sentence),
                    on_status=lambda status: self.progress.emit(status),
                    should_stop=self._should_stop,
                )

                response = str(response or "").strip()

                if self._stop_requested:
                    break

                if not response:
                    response = "I did not generate a clear answer."

                self.ai_response.emit(response)

                if self._stop_requested:
                    break

                self.progress.emit("✅ Ready for next question.")
                time.sleep(self.pause_after_speech)

            self.progress.emit("Live chat stopped.")
            self.completed.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class AnalyzeImageWorker(QThread):
    progress = Signal(str)
    completed = Signal(str, str)
    error = Signal(str)

    def __init__(
        self,
        runtime: ModelRuntime,
        image_path: Path,
        question: str,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.image_path = Path(image_path)
        self.question = question

    def run(self) -> None:
        try:
            self.progress.emit("Analyzing image with Qwen-VL...")

            vision_answer, spoken_answer = self.runtime.analyze_image_safe_and_speak(
                image_path=self.image_path,
                question=self.question,
            )

            self.progress.emit("Done.")
            self.completed.emit(vision_answer, spoken_answer)

        except Exception as exc:
            self.error.emit(str(exc))


class SpeakOnlyWorker(QThread):
    progress = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(self, runtime: ModelRuntime, text: str) -> None:
        super().__init__()
        self.runtime = runtime
        self.text = text

    def run(self) -> None:
        try:
            self.progress.emit("XTTSFast is speaking...")

            self.runtime.speak(self.text)

            self.progress.emit("Done.")
            self.completed.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class RecordReferenceWorker(QThread):
    progress = Signal(str)
    completed = Signal()
    error = Signal(str)

    def __init__(self, runtime: ModelRuntime, seconds: int = 8) -> None:
        super().__init__()
        self.runtime = runtime
        self.seconds = seconds

    def run(self) -> None:
        try:
            self.progress.emit(f"Recording reference voice for {self.seconds}s...")

            self.runtime.xtts.record_reference(seconds=self.seconds)

            self.progress.emit("Reference voice recorded.")
            self.completed.emit()

        except Exception as exc:
            self.error.emit(str(exc))




class WebKnowledgeWorker(QThread):
    progress = Signal(str)
    completed = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        mode: str,
        value: str = "",
        max_results: int = 3,
        rebuild: bool = True,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.value = str(value or "").strip()
        self.max_results = int(max_results)
        self.rebuild = bool(rebuild)

    def run(self) -> None:
        try:
            from core.web_knowledge import WebKnowledgeIngestor

            ingestor = WebKnowledgeIngestor(root_dir=ROOT_DIR)

            lines: list[str] = []

            if self.mode == "search":
                if not self.value:
                    raise ValueError("Web search query is empty.")

                self.progress.emit("🌐 Searching web...")

                documents, errors = ingestor.search_and_ingest(
                    query=self.value,
                    max_results=self.max_results,
                )

                lines.append(f"Web search query: {self.value}")
                lines.append(f"Saved documents: {len(documents)}")

                for index, document in enumerate(documents, start=1):
                    lines.append(
                        f"{index}. {document.title}\n"
                        f"   URL: {document.url}\n"
                        f"   Path: {document.saved_path}\n"
                        f"   Text chars: {document.text_chars}"
                    )

                if errors:
                    lines.append("")
                    lines.append("Pages skipped or failed:")

                    for error in errors[:5]:
                        lines.append(f"- {error}")

            elif self.mode == "url":
                if not self.value:
                    raise ValueError("URL is empty.")

                self.progress.emit("🌐 Crawling URL...")

                document = ingestor.crawl_and_save(self.value)

                lines.append("Crawled URL successfully.")
                lines.append(f"Title: {document.title}")
                lines.append(f"URL: {document.url}")
                lines.append(f"Saved path: {document.saved_path}")
                lines.append(f"Text chars: {document.text_chars}")

            elif self.mode == "rebuild":
                self.progress.emit("🔁 Rebuilding RAG...")
                output = ingestor.rebuild_rag_index()
                lines.append("RAG rebuild completed.")
                lines.append(output)

            else:
                raise ValueError(f"Unsupported web knowledge mode: {self.mode}")

            if self.rebuild and self.mode in {"search", "url"}:
                self.progress.emit("🔁 Rebuilding RAG...")
                output = ingestor.rebuild_rag_index()
                lines.append("")
                lines.append("RAG rebuild completed.")
                lines.append(output)

            self.progress.emit("✅ Web knowledge ready.")
            self.completed.emit("\n".join(lines).strip())

        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.runtime: ModelRuntime | None = None
        self.settings = AppSettings.load(ROOT_DIR)
        self.selected_image_path: Path | None = None

        self.live_worker: LiveChatWorker | None = None
        self.live_chat_active = False
        self._live_ai_stream_open = False

        self.setWindowTitle("🍓 LocalAI V3")
        self.resize(1220, 920)
        self.setMinimumSize(1000, 760)

        self._build_ui()
        self._apply_background_overlay_theme()
        self._set_runtime_ready(False)

        if self.settings.auto_load_on_start:
            QTimer.singleShot(500, self._load_runtime)

    def _make_label(
        self,
        text: str,
        object_name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        align: Qt.AlignmentFlag = Qt.AlignLeft | Qt.AlignVCenter,
    ) -> QLabel:
        label = QLabel(text, self.canvas)
        label.setObjectName(object_name)
        label.setAlignment(align)
        label.setGeometry(x, y, w, h)
        return label

    def _button(self, text: str, tooltip: str = "") -> QPushButton:
        button = QPushButton(text, self.canvas)
        button.setObjectName("overlayButton")
        button.setToolTip(tooltip or text)
        button.setCursor(Qt.PointingHandCursor)
        return button

    def _build_ui(self) -> None:
        bg_path = ROOT_DIR / "assets" / "ui" / "cute_ui_bg.png"

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setObjectName("appScroll")
        self.setCentralWidget(scroll)

        self.canvas = BackgroundCanvas(bg_path=bg_path)
        scroll.setWidget(self.canvas)

        self._make_label("🌸 LocalAI V3", "titleLabel", 248, 72, 380, 58)
        self._make_label("English-only local voice assistant", "subtitleLabel", 253, 126, 420, 32)

        self.status_label = QLabel("Runtime not loaded.", self.canvas)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setGeometry(725, 76, 205, 48)

        self._make_label(
            "💬  User input / ASR result / image question",
            "inputTitle",
            55,
            176,
            500,
            36,
        )

        self.input_box = QTextEdit(self.canvas)
        self.input_box.setObjectName("inputBox")
        self.input_box.setGeometry(51, 228, 575, 205)
        self.input_box.setPlaceholderText("Type here or speak...")
        self.input_box.setPlainText(self.settings.default_user_text)

        self._make_label(
            "✨  English response / live chat",
            "responseTitle",
            55,
            470,
            450,
            36,
        )

        self.response_box = QTextEdit(self.canvas)
        self.response_box.setObjectName("responseBox")
        self.response_box.setReadOnly(True)
        self.response_box.setGeometry(51, 530, 575, 165)
        self.response_box.setPlaceholderText("English response will appear here...")

        self.choose_image_button = self._button("🖼️ Choose Image", "Choose image for Qwen-VL")
        self.choose_image_button.setGeometry(150, 748, 145, 42)
        self.choose_image_button.clicked.connect(self._choose_image)

        self.image_label = QLabel("Selected: no image yet", self.canvas)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.image_label.setGeometry(305, 750, 210, 38)

        self.clear_image_button = self._button("🧹 Clear Image", "Clear selected image")
        self.clear_image_button.setGeometry(520, 748, 120, 42)
        self.clear_image_button.clicked.connect(self._clear_image)

        self._make_label("🎀 Controls", "controlsTitle", 690, 176, 260, 36)

        self.load_button = self._button("☁️ Load All", "Load all models")
        self.load_button.setGeometry(684, 218, 220, 44)
        self.load_button.clicked.connect(self._load_runtime)

        self.ask_speak_button = self._button("💬 Ask & Speak", "Ask Gemma and speak with XTTS")
        self.ask_speak_button.setGeometry(925, 218, 220, 44)
        self.ask_speak_button.clicked.connect(self._ask_and_speak)

        self.voice_button = self._button("🎙️ Voice Question", "Auto voice question with VAD")
        self.voice_button.setGeometry(684, 276, 220, 44)
        self.voice_button.clicked.connect(self._voice_question)

        self.speak_response_button = self._button("🔊 Speak Again", "Speak response again")
        self.speak_response_button.setGeometry(925, 276, 220, 44)
        self.speak_response_button.clicked.connect(self._speak_response_again)

        self.record_button = self._button("⭕ Record Voice", "Record XTTS reference voice")
        self.record_button.setGeometry(684, 334, 220, 44)
        self.record_button.clicked.connect(self._record_reference)

        self.clear_history_button = self._button("🧺 Clear History", "Clear conversation history")
        self.clear_history_button.setGeometry(925, 334, 220, 44)
        self.clear_history_button.clicked.connect(self._clear_history)

        self.live_start_button = self._button("🟢 Start Live", "Start continuous live voice chat")
        self.live_start_button.setGeometry(684, 392, 220, 44)
        self.live_start_button.clicked.connect(self._start_live_chat)

        self.live_stop_button = self._button("🔴 Stop Live", "Stop continuous live voice chat")
        self.live_stop_button.setGeometry(925, 392, 220, 44)
        self.live_stop_button.clicked.connect(self._stop_live_chat)

        self.stop_audio_button = self._button("🛑 Stop Audio", "Stop current audio playback")
        self.stop_audio_button.setGeometry(684, 450, 220, 44)
        self.stop_audio_button.clicked.connect(self._stop_audio)

        self.analyze_image_button = self._button("🔎 Analyze Image", "Analyze selected image and speak result")
        self.analyze_image_button.setGeometry(925, 450, 220, 44)
        self.analyze_image_button.clicked.connect(self._analyze_image_and_speak)

        self._make_label("🍓 Settings", "settingsTitle", 690, 520, 260, 36)

        self._make_label("⏱️ Max voice seconds", "settingLabel", 690, 568, 230, 34)
        self.voice_seconds_spin = QSpinBox(self.canvas)
        self.voice_seconds_spin.setObjectName("overlaySpin")
        self.voice_seconds_spin.setRange(2, 60)
        self.voice_seconds_spin.setValue(self.settings.voice_question_seconds)
        self.voice_seconds_spin.setAlignment(Qt.AlignCenter)
        self.voice_seconds_spin.setGeometry(925, 568, 190, 34)

        self._make_label("⏱️ Reference seconds", "settingLabel", 690, 608, 230, 34)
        self.reference_seconds_spin = QSpinBox(self.canvas)
        self.reference_seconds_spin.setObjectName("overlaySpin")
        self.reference_seconds_spin.setRange(4, 20)
        self.reference_seconds_spin.setValue(self.settings.reference_voice_seconds)
        self.reference_seconds_spin.setAlignment(Qt.AlignCenter)
        self.reference_seconds_spin.setGeometry(925, 608, 190, 34)

        self._make_label("🎭 Personality", "settingLabel", 690, 648, 190, 34)

        self.personality_combo = QComboBox(self.canvas)
        self.personality_combo.setObjectName("overlayCombo")
        self.personality_combo.setGeometry(830, 648, 285, 34)

        for key, display_name in get_persona_options():
            self.personality_combo.addItem(display_name, key)

        current_index = self.personality_combo.findData(self.settings.personality_mode)

        if current_index >= 0:
            self.personality_combo.setCurrentIndex(current_index)

        self.persona_checkbox = QCheckBox("Persona", self.canvas)
        self.persona_checkbox.setObjectName("overlayCheck")
        self.persona_checkbox.setChecked(self.settings.personality_enabled)
        self.persona_checkbox.setGeometry(688, 686, 92, 34)

        self.emotion_checkbox = QCheckBox("Emotion", self.canvas)
        self.emotion_checkbox.setObjectName("overlayCheck")
        self.emotion_checkbox.setChecked(self.settings.emotion_enabled)
        self.emotion_checkbox.setGeometry(780, 686, 92, 34)

        self.memory_checkbox = QCheckBox("Memory", self.canvas)
        self.memory_checkbox.setObjectName("overlayCheck")
        self.memory_checkbox.setChecked(self.settings.memory_enabled)
        self.memory_checkbox.setGeometry(875, 686, 92, 34)

        self.rag_checkbox = QCheckBox("RAG", self.canvas)
        self.rag_checkbox.setObjectName("overlayCheck")
        self.rag_checkbox.setChecked(self.settings.rag_enabled)
        self.rag_checkbox.setGeometry(970, 686, 65, 34)

        self.auto_load_checkbox = QCheckBox("Auto", self.canvas)
        self.auto_load_checkbox.setObjectName("overlayCheck")
        self.auto_load_checkbox.setChecked(self.settings.auto_load_on_start)
        self.auto_load_checkbox.setGeometry(1035, 686, 70, 34)

        self.save_settings_button = self._button("🌼 Save", "Save settings")
        self.save_settings_button.setGeometry(1098, 686, 52, 36)
        self.save_settings_button.clicked.connect(self._save_settings)

        self._make_label("📒 Log", "logTitle", 690, 728, 260, 34)

        self.log_box = QTextEdit(self.canvas)
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setGeometry(690, 770, 455, 78)
        self.log_box.setPlaceholderText("Logs will appear here...")

        self._make_label("🌐 Web Knowledge", "webTitle", 690, 858, 260, 30)

        self.web_input = QLineEdit(self.canvas)
        self.web_input.setObjectName("overlayLine")
        self.web_input.setGeometry(690, 892, 455, 34)
        self.web_input.setPlaceholderText("Enter search query or direct URL...")

        self.web_search_button = self._button("🔍 Search", "Search web and save pages into RAG")
        self.web_search_button.setGeometry(690, 934, 108, 36)
        self.web_search_button.clicked.connect(self._web_search_to_rag)

        self.web_crawl_button = self._button("🌐 Crawl URL", "Crawl one direct URL into RAG")
        self.web_crawl_button.setGeometry(806, 934, 118, 36)
        self.web_crawl_button.clicked.connect(self._crawl_url_to_rag)

        self.web_rebuild_button = self._button("🔁 Rebuild", "Rebuild RAG index")
        self.web_rebuild_button.setGeometry(932, 934, 100, 36)
        self.web_rebuild_button.clicked.connect(self._rebuild_rag_index)

        self.web_results_spin = QSpinBox(self.canvas)
        self.web_results_spin.setObjectName("overlaySpin")
        self.web_results_spin.setRange(1, 5)
        self.web_results_spin.setValue(3)
        self.web_results_spin.setAlignment(Qt.AlignCenter)
        self.web_results_spin.setGeometry(1040, 934, 50, 36)
        self.web_results_spin.setToolTip("Maximum web pages to crawl from search results")

        self.web_auto_rebuild_checkbox = QCheckBox("RAG", self.canvas)
        self.web_auto_rebuild_checkbox.setObjectName("overlayCheck")
        self.web_auto_rebuild_checkbox.setChecked(True)
        self.web_auto_rebuild_checkbox.setGeometry(1095, 934, 55, 36)
        self.web_auto_rebuild_checkbox.setToolTip("Automatically rebuild RAG after crawling")

    def _apply_background_overlay_theme(self) -> None:
        self.setStyleSheet(
            """
            QScrollArea#appScroll {
                border: none;
                background: #fff3f8;
            }

            QWidget#backgroundCanvas {
                background: #fff8fc;
            }

            QLabel#titleLabel {
                background: transparent;
                color: #ff5f9c;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 34px;
                font-weight: 900;
            }

            QLabel#subtitleLabel {
                background: transparent;
                color: #9275aa;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 15px;
                font-weight: 800;
            }

            QLabel#inputTitle,
            QLabel#responseTitle,
            QLabel#controlsTitle,
            QLabel#settingsTitle,
            QLabel#logTitle,
            QLabel#webTitle {
                background: transparent;
                color: #f05f9b;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 17px;
                font-weight: 900;
            }

            QLabel#responseTitle,
            QLabel#logTitle,
            QLabel#webTitle {
                color: #9b72e8;
            }

            QLabel#settingLabel {
                background: transparent;
                color: #55445f;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 14px;
                font-weight: 900;
            }

            QLabel#statusLabel {
                background: rgba(232, 255, 223, 225);
                color: #377b34;
                border: 2px solid rgba(191, 233, 184, 210);
                border-radius: 22px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 15px;
                font-weight: 900;
            }

            QTextEdit#inputBox,
            QTextEdit#responseBox {
                background: rgba(255, 255, 255, 218);
                color: #55445f;
                border: 1px solid rgba(255, 190, 220, 120);
                border-radius: 17px;
                padding: 12px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 14px;
                font-weight: 500;
                selection-background-color: #ffd1e8;
                selection-color: #55445f;
            }

            QTextEdit#inputBox:focus {
                background: rgba(255, 255, 255, 240);
                border: 2px solid rgba(255, 143, 196, 190);
            }

            QTextEdit#responseBox {
                border: 1px solid rgba(185, 159, 255, 120);
            }

            QTextEdit#responseBox:focus {
                background: rgba(255, 255, 255, 240);
                border: 2px solid rgba(185, 159, 255, 190);
            }

            QLabel#imageLabel {
                background: rgba(243, 250, 255, 150);
                color: #2d63c8;
                border: none;
                border-radius: 12px;
                padding-left: 8px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
                font-weight: 900;
            }

            QTextEdit#logBox {
                background: rgba(255, 255, 255, 220);
                color: #4f4458;
                border: 1px solid rgba(216, 202, 255, 120);
                border-radius: 13px;
                padding: 8px;
                font-family: "Consolas", "Segoe UI", monospace;
                font-size: 11px;
                font-weight: 600;
            }

            QPushButton#overlayButton {
                background: rgba(255, 255, 255, 130);
                color: #3f334a;
                border: 1px solid rgba(255, 160, 200, 95);
                border-radius: 17px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
                font-weight: 900;
            }

            QPushButton#overlayButton:hover {
                background: rgba(255, 245, 250, 220);
                border: 2px solid rgba(255, 143, 195, 170);
            }

            QPushButton#overlayButton:pressed {
                background: rgba(255, 210, 231, 180);
                border: 2px solid rgba(255, 111, 174, 190);
                padding-top: 3px;
            }

            QPushButton#overlayButton:disabled {
                background: rgba(240, 235, 241, 120);
                color: rgba(120, 110, 125, 145);
                border: 1px solid rgba(220, 210, 225, 120);
            }


            QLineEdit#overlayLine {
                background: rgba(255, 255, 255, 220);
                color: #55445f;
                border: 1px solid rgba(185, 159, 255, 120);
                border-radius: 13px;
                padding-left: 10px;
                padding-right: 10px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
                font-weight: 800;
                selection-background-color: #ffd1e8;
                selection-color: #55445f;
            }

            QLineEdit#overlayLine:focus {
                background: rgba(255, 255, 255, 240);
                border: 2px solid rgba(185, 159, 255, 190);
            }

            QSpinBox#overlaySpin {
                background: rgba(255, 255, 255, 220);
                color: #55445f;
                border: 1px solid rgba(255, 160, 200, 120);
                border-radius: 13px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 14px;
                font-weight: 900;
            }

            QSpinBox#overlaySpin:focus {
                border: 2px solid rgba(255, 143, 195, 190);
            }

            QComboBox#overlayCombo {
                background: rgba(255, 255, 255, 220);
                color: #55445f;
                border: 1px solid rgba(255, 160, 200, 120);
                border-radius: 13px;
                padding-left: 10px;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
                font-weight: 900;
            }

            QComboBox#overlayCombo:focus {
                border: 2px solid rgba(255, 143, 195, 190);
            }

            QComboBox#overlayCombo::drop-down {
                border: none;
                width: 28px;
            }

            QCheckBox#overlayCheck {
                background: rgba(255, 255, 255, 0);
                color: #55445f;
                font-family: "Segoe UI", "Microsoft YaHei UI", Arial;
                font-size: 13px;
                font-weight: 900;
                spacing: 8px;
            }

            QCheckBox#overlayCheck::indicator {
                width: 20px;
                height: 20px;
                border-radius: 7px;
                border: 2px solid #ff9fbd;
                background: #ffffff;
            }

            QCheckBox#overlayCheck::indicator:checked {
                background: #ff6fae;
                border: 2px solid #ff5a9f;
            }
            """
        )

    def _load_runtime(self) -> None:
        self._set_all_action_buttons(False)

        self.status_label.setText("🟡 Loading...")
        self._log("INFO  Loading Gemma, XTTSFast, and ASR...")

        mode = self._current_personality_mode()
        personality_enabled = bool(self.persona_checkbox.isChecked())
        emotion_enabled = bool(self.emotion_checkbox.isChecked())
        memory_enabled = bool(self.memory_checkbox.isChecked())
        rag_enabled = bool(self.rag_checkbox.isChecked())

        self.load_worker = LoadRuntimeWorker(
            personality_mode=mode,
            personality_enabled=personality_enabled,
            emotion_enabled=emotion_enabled,
            memory_enabled=memory_enabled,
            rag_enabled=rag_enabled,
        )
        self.load_worker.progress.connect(self._on_progress)
        self.load_worker.loaded.connect(self._on_runtime_loaded)
        self.load_worker.error.connect(self._on_error)
        self.load_worker.start()

    def _on_runtime_loaded(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime

        self.status_label.setText("🟢 Runtime ready")
        self._log("INFO  Runtime ready.")
        self._log(f"INFO  Personality mode: {self.runtime.get_personality_mode()}")
        self._log(f"INFO  Personality enabled: {self.runtime.get_personality_enabled()}")
        self._log(f"INFO  Emotion state: {self.runtime.get_emotion_summary()}")
        self._log(f"INFO  Memory state: {self.runtime.get_memory_summary()}")
        self._log(f"INFO  RAG state: {self.runtime.get_rag_summary()}")

        self._set_runtime_ready(True)

    def _ask_and_speak(self) -> None:
        if self.runtime is None:
            return

        self._sync_runtime_personality()

        text = self.input_box.toPlainText().strip()

        if not text:
            self._log("WARN  No user input.")
            return

        self.response_box.clear()

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)

        self.status_label.setText("💭 Thinking...")
        self._log(f"USER  {text}")

        self.ask_worker = AskSpeakWorker(self.runtime, text)
        self.ask_worker.progress.connect(self._on_progress)
        self.ask_worker.ai_response.connect(self._on_ai_response)
        self.ask_worker.completed.connect(self._on_action_done)
        self.ask_worker.error.connect(self._on_error)
        self.ask_worker.start()

    def _voice_question(self) -> None:
        if self.runtime is None:
            return

        self._sync_runtime_personality()

        seconds = int(self.voice_seconds_spin.value())

        self.input_box.clear()
        self.response_box.clear()

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)

        self.status_label.setText("🎙️ Listening...")
        self._log(f"INFO  Auto listening. Max voice time: {seconds} seconds.")

        self.voice_worker = VoiceAssistantWorker(self.runtime, seconds=seconds)
        self.voice_worker.progress.connect(self._on_progress)
        self.voice_worker.user_text.connect(self._on_user_text)
        self.voice_worker.ai_response.connect(self._on_ai_response)
        self.voice_worker.completed.connect(self._on_action_done)
        self.voice_worker.error.connect(self._on_error)
        self.voice_worker.start()

    def _start_live_chat(self) -> None:
        if self.runtime is None:
            return

        if self.live_chat_active:
            self._log("WARN  Live chat is already running.")
            return

        self._sync_runtime_personality()

        seconds = int(self.voice_seconds_spin.value())

        self.live_chat_active = True

        self.response_box.clear()
        self.response_box.append("Live chat started.\n")

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)
        self.live_start_button.setEnabled(False)
        self.live_stop_button.setEnabled(True)
        self.stop_audio_button.setEnabled(True)

        self.status_label.setText("🟢 Live chat")
        self._log(f"INFO  Live chat started. Max listen time per turn: {seconds}s.")

        self.live_worker = LiveChatWorker(
            runtime=self.runtime,
            seconds=seconds,
            pause_after_speech=0.25,
        )

        self.live_worker.progress.connect(self._on_progress)
        self.live_worker.user_text.connect(self._on_live_user_text)
        self.live_worker.partial_response.connect(self._on_live_partial_response)
        self.live_worker.speech_sentence.connect(self._on_live_speech_sentence)
        self.live_worker.ai_response.connect(self._on_live_ai_response)
        self.live_worker.completed.connect(self._on_live_chat_done)
        self.live_worker.error.connect(self._on_error)
        self.live_worker.start()

    def _stop_live_chat(self) -> None:
        if not self.live_chat_active:
            return

        self.status_label.setText("🔴 Stopping live...")
        self._log("INFO  Stop live chat requested.")

        if self.live_worker is not None:
            self.live_worker.request_stop()

        try:
            sd.stop()
        except Exception:
            pass

        self.live_stop_button.setEnabled(False)
        self.stop_audio_button.setEnabled(True)

    def _on_live_user_text(self, text: str) -> None:
        self.input_box.setPlainText(text)
        self.response_box.moveCursor(QTextCursor.End)
        self.response_box.insertPlainText(f"\nUSER: {text}\nAI: ")
        self._live_ai_stream_open = True
        self._log(f"LIVE USER  {text}")

    def _on_live_partial_response(self, token: str) -> None:
        token = str(token or "")

        if not token:
            return

        self.response_box.moveCursor(QTextCursor.End)
        self.response_box.insertPlainText(token)
        self.response_box.moveCursor(QTextCursor.End)

    def _on_live_speech_sentence(self, sentence: str) -> None:
        sentence = str(sentence or "").strip()

        if sentence:
            self._log(f"LIVE SPEAK  {sentence}")

    def _on_live_ai_response(self, response: str) -> None:
        response = str(response or "").strip()

        if self._live_ai_stream_open:
            self.response_box.moveCursor(QTextCursor.End)
            self.response_box.insertPlainText("\n")
            self._live_ai_stream_open = False
        else:
            self.response_box.append(f"AI: {response}\n")

        self._log(f"LIVE AI    {response}")

    def _on_live_chat_done(self) -> None:
        self.live_chat_active = False
        self.live_worker = None
        self._live_ai_stream_open = False

        self.status_label.setText("🟢 Runtime ready")
        self.response_box.append("Live chat stopped.")
        self._log("INFO  Live chat stopped.")

        self._set_runtime_ready(True)

    def _choose_image(self) -> None:

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            str(Path.home() / "Desktop"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )

        if not file_path:
            return

        self.selected_image_path = Path(file_path)

        self.image_label.setText(f"Selected: {self.selected_image_path.name}")
        self.status_label.setText("🖼️ Image selected")
        self._log(f"USER  Selected image: {self.selected_image_path.name}")

        if not self.input_box.toPlainText().strip():
            self.input_box.setPlainText("What is in this image?")

    def _clear_image(self) -> None:
        self.selected_image_path = None
        self.image_label.setText("Selected: no image yet")

        current_text = self.input_box.toPlainText().strip().lower()

        if current_text in {
            "what is in this image?",
            "describe this image.",
            "describe this image",
        }:
            self.input_box.clear()

        self.status_label.setText("🖼️ Image cleared")
        self._log("INFO  Selected image cleared.")

    def _analyze_image_and_speak(self) -> None:
        if self.runtime is None:
            return

        self._sync_runtime_personality()

        if self.selected_image_path is None:
            self._log("WARN  No image selected.")
            self.status_label.setText("No image selected")
            return

        question = self.input_box.toPlainText().strip()

        if not question:
            question = "What is in this image?"

        self.response_box.clear()

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)

        self.status_label.setText("🔎 Analyzing...")
        self._log(f"USER  Image question: {question}")
        self._log(f"INFO  Image path: {self.selected_image_path}")

        self.image_worker = AnalyzeImageWorker(
            runtime=self.runtime,
            image_path=self.selected_image_path,
            question=question,
        )
        self.image_worker.progress.connect(self._on_progress)
        self.image_worker.completed.connect(self._on_image_analysis_done)
        self.image_worker.error.connect(self._on_error)
        self.image_worker.start()

    def _on_image_analysis_done(self, vision_answer: str, spoken_answer: str) -> None:
        text = (
            "Qwen-VL vision answer:\n"
            f"{vision_answer}\n\n"
            "Spoken answer:\n"
            f"{spoken_answer}"
        )

        self.response_box.setPlainText(text)

        self._log(f"INFO  Qwen-VL: {vision_answer}")
        self._log(f"INFO  Spoken: {spoken_answer}")

        self.status_label.setText("🟢 Runtime ready")
        self._log("INFO  Image analysis finished.")

        self._set_runtime_ready(True)

    def _speak_response_again(self) -> None:
        if self.runtime is None:
            return

        text = self.response_box.toPlainText().strip()

        if not text:
            self._log("WARN  No response to speak.")
            return

        if "Spoken answer:" in text:
            text = text.split("Spoken answer:", 1)[1].strip()

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)

        self.status_label.setText("🔊 Speaking...")
        self._log(f"INFO  Speaking again: {text}")

        self.speak_worker = SpeakOnlyWorker(self.runtime, text)
        self.speak_worker.progress.connect(self._on_progress)
        self.speak_worker.completed.connect(self._on_action_done)
        self.speak_worker.error.connect(self._on_error)
        self.speak_worker.start()

    def _record_reference(self) -> None:
        if self.runtime is None:
            return

        seconds = int(self.reference_seconds_spin.value())

        self._set_all_action_buttons(False)
        self._set_stop_audio_enabled(True)

        self.status_label.setText("🎀 Recording...")
        self._log(f"INFO  Recording reference voice for {seconds} seconds.")

        self.record_worker = RecordReferenceWorker(self.runtime, seconds=seconds)
        self.record_worker.progress.connect(self._on_progress)
        self.record_worker.completed.connect(self._on_action_done)
        self.record_worker.error.connect(self._on_error)
        self.record_worker.start()

    def _web_search_to_rag(self) -> None:
        query = self.web_input.text().strip()

        if not query:
            self._log("WARN  Web search query is empty.")
            self.status_label.setText("No web query")
            return

        if query.lower().startswith(("http://", "https://")):
            self._log("INFO  Input looks like a URL. Use Crawl URL for direct pages.")

        self._start_web_knowledge_worker(
            mode="search",
            value=query,
            max_results=int(self.web_results_spin.value()),
            rebuild=bool(self.web_auto_rebuild_checkbox.isChecked()),
        )

    def _crawl_url_to_rag(self) -> None:
        url = self.web_input.text().strip()

        if not url:
            self._log("WARN  URL is empty.")
            self.status_label.setText("No URL")
            return

        if not url.lower().startswith(("http://", "https://")):
            self._log("WARN  Please enter a full URL starting with http:// or https://")
            self.status_label.setText("Invalid URL")
            return

        self._start_web_knowledge_worker(
            mode="url",
            value=url,
            max_results=1,
            rebuild=bool(self.web_auto_rebuild_checkbox.isChecked()),
        )

    def _rebuild_rag_index(self) -> None:
        self._start_web_knowledge_worker(
            mode="rebuild",
            value="",
            max_results=1,
            rebuild=False,
        )

    def _start_web_knowledge_worker(
        self,
        mode: str,
        value: str,
        max_results: int,
        rebuild: bool,
    ) -> None:
        if self.live_chat_active:
            self._log("WARN  Stop Live Chat before changing the RAG knowledge base.")
            return

        self._set_all_action_buttons(False)
        self._set_web_buttons(False)

        if mode == "search":
            self.status_label.setText("🌐 Searching...")
            self._log(f"WEB   Search query: {value}")
        elif mode == "url":
            self.status_label.setText("🌐 Crawling...")
            self._log(f"WEB   Crawl URL: {value}")
        else:
            self.status_label.setText("🔁 Rebuilding...")
            self._log("WEB   Rebuild RAG index requested.")

        self.web_worker = WebKnowledgeWorker(
            mode=mode,
            value=value,
            max_results=max_results,
            rebuild=rebuild,
        )
        self.web_worker.progress.connect(self._on_progress)
        self.web_worker.completed.connect(self._on_web_knowledge_done)
        self.web_worker.error.connect(self._on_web_knowledge_error)
        self.web_worker.start()

    def _on_web_knowledge_done(self, summary: str) -> None:
        summary = str(summary or "Web knowledge task completed.").strip()

        self.response_box.setPlainText(
            "Web Knowledge Result:\n\n"
            f"{summary}\n\n"
            "Next step: turn on the RAG checkbox and ask questions about the crawled content."
        )

        self._log("WEB   Task completed.")

        for line in summary.splitlines()[:12]:
            self._log(f"WEB   {line}")

        if self.runtime is not None:
            try:
                self._log(f"INFO  RAG state: {self.runtime.get_rag_summary()}")
            except Exception:
                pass

        self.status_label.setText("🟢 Web ready")
        self._set_runtime_ready(self.runtime is not None)
        self._set_web_buttons(True)

    def _on_web_knowledge_error(self, error: str) -> None:
        self.status_label.setText("🔴 Web error")
        self._log(f"WEB ERROR {error}")
        self.response_box.setPlainText(
            "Web Knowledge Error:\n\n"
            f"{error}\n\n"
            "Try a direct URL, use a different public page, or rebuild RAG manually."
        )
        self._set_runtime_ready(self.runtime is not None)
        self._set_web_buttons(True)


    def _current_personality_mode(self) -> str:
        try:
            mode = self.personality_combo.currentData()
            return str(mode or "project_engineer")
        except Exception:
            return "project_engineer"

    def _sync_runtime_personality(self) -> None:
        if self.runtime is None:
            return

        mode = self._current_personality_mode()
        personality_enabled = bool(self.persona_checkbox.isChecked())
        emotion_enabled = bool(self.emotion_checkbox.isChecked())
        memory_enabled = bool(self.memory_checkbox.isChecked())
        rag_enabled = bool(self.rag_checkbox.isChecked())

        self.runtime.set_personality_mode(mode)
        self.runtime.set_personality_enabled(personality_enabled)
        self.runtime.set_emotion_enabled(emotion_enabled)
        self.runtime.set_memory_enabled(memory_enabled)
        self.runtime.set_rag_enabled(rag_enabled)

        self._log(f"INFO  Personality mode: {mode}")
        self._log(f"INFO  Personality enabled: {personality_enabled}")
        self._log(f"INFO  Emotion enabled: {emotion_enabled}")
        self._log(f"INFO  Memory enabled: {memory_enabled}")
        self._log(f"INFO  RAG enabled: {rag_enabled}")
        self._log(f"INFO  Emotion state: {self.runtime.get_emotion_summary()}")

        if memory_enabled:
            self._log(f"INFO  Memory state: {self.runtime.get_memory_summary()}")

        if rag_enabled:
            self._log(f"INFO  RAG state: {self.runtime.get_rag_summary()}")

    def _save_settings(self) -> None:
        self.settings.voice_question_seconds = int(self.voice_seconds_spin.value())
        self.settings.reference_voice_seconds = int(self.reference_seconds_spin.value())
        self.settings.default_user_text = (
            self.input_box.toPlainText().strip() or "What can you do for me?"
        )
        self.settings.auto_load_on_start = bool(self.auto_load_checkbox.isChecked())
        self.settings.personality_mode = self._current_personality_mode()
        self.settings.personality_enabled = bool(self.persona_checkbox.isChecked())
        self.settings.emotion_enabled = bool(self.emotion_checkbox.isChecked())
        self.settings.memory_enabled = bool(self.memory_checkbox.isChecked())
        self.settings.rag_enabled = bool(self.rag_checkbox.isChecked())

        self._sync_runtime_personality()

        settings_path = self.settings.save(ROOT_DIR)

        self.status_label.setText("💾 Saved")
        self._log(f"INFO  Settings saved: {settings_path}")

    def _clear_history(self) -> None:
        if self.runtime is None:
            return

        self.runtime.reset_conversation()
        self.response_box.clear()

        self.status_label.setText("🧹 Cleared")
        self._log("INFO  Conversation history and emotion cleared.")

    def _stop_audio(self) -> None:
        try:
            sd.stop()
            self.status_label.setText("🛑 Audio stopped")
            self._log("INFO  Audio stopped.")
        except Exception as exc:
            self._log(f"ERROR Stop audio failed: {exc}")

    def _on_user_text(self, text: str) -> None:
        self.input_box.setPlainText(text)
        self._log(f"ASR   {text}")

    def _on_ai_response(self, response: str) -> None:
        self.response_box.setPlainText(response)
        self._log(f"AI    {response}")

    def _on_action_done(self) -> None:
        self.status_label.setText("🟢 Runtime ready")
        self._log("INFO  Ready.")

        if self.runtime is not None:
            self._log(f"INFO  Emotion state: {self.runtime.get_emotion_summary()}")

            try:
                self._log(f"INFO  Memory state: {self.runtime.get_memory_summary()}")
                self._log(f"INFO  RAG state: {self.runtime.get_rag_summary()}")
            except Exception:
                pass

        self._set_runtime_ready(True)

    def _on_progress(self, text: str) -> None:
        self.status_label.setText(text[:28])
        self._log(f"INFO  {text}")

    def _on_error(self, error: str) -> None:
        self.live_chat_active = False

        self.status_label.setText("🔴 Error")
        self._log(f"ERROR {error}")

        self._set_runtime_ready(self.runtime is not None)

    def _set_runtime_ready(self, ready: bool) -> None:
        self.load_button.setEnabled(not ready)

        self.ask_speak_button.setEnabled(ready and not self.live_chat_active)
        self.voice_button.setEnabled(ready and not self.live_chat_active)
        self.speak_response_button.setEnabled(ready and not self.live_chat_active)
        self.record_button.setEnabled(ready and not self.live_chat_active)
        self.clear_history_button.setEnabled(ready and not self.live_chat_active)

        self.live_start_button.setEnabled(ready and not self.live_chat_active)
        self.live_stop_button.setEnabled(ready and self.live_chat_active)

        self.stop_audio_button.setEnabled(ready)
        self.choose_image_button.setEnabled(ready and not self.live_chat_active)
        self.clear_image_button.setEnabled(True)
        self.analyze_image_button.setEnabled(ready and not self.live_chat_active)

        self.save_settings_button.setEnabled(True)
        self.voice_seconds_spin.setEnabled(True)
        self.reference_seconds_spin.setEnabled(True)
        self.personality_combo.setEnabled(True)
        self.persona_checkbox.setEnabled(True)
        self.emotion_checkbox.setEnabled(True)
        self.memory_checkbox.setEnabled(True)
        self.rag_checkbox.setEnabled(True)
        self.auto_load_checkbox.setEnabled(True)
        self._set_web_buttons(ready and not self.live_chat_active)

    def _set_all_action_buttons(self, enabled: bool) -> None:
        self.load_button.setEnabled(enabled)
        self.ask_speak_button.setEnabled(enabled)
        self.voice_button.setEnabled(enabled)
        self.speak_response_button.setEnabled(enabled)
        self.record_button.setEnabled(enabled)
        self.clear_history_button.setEnabled(enabled)

        self.live_start_button.setEnabled(enabled and not self.live_chat_active)
        self.live_stop_button.setEnabled(self.live_chat_active)

        self.stop_audio_button.setEnabled(self.runtime is not None)
        self.choose_image_button.setEnabled(enabled)
        self.clear_image_button.setEnabled(True)
        self.analyze_image_button.setEnabled(enabled)

        self.save_settings_button.setEnabled(True)
        self.voice_seconds_spin.setEnabled(True)
        self.reference_seconds_spin.setEnabled(True)
        self.personality_combo.setEnabled(True)
        self.persona_checkbox.setEnabled(True)
        self.emotion_checkbox.setEnabled(True)
        self.memory_checkbox.setEnabled(True)
        self.rag_checkbox.setEnabled(True)
        self.auto_load_checkbox.setEnabled(True)
        self._set_web_buttons(enabled and not self.live_chat_active)

    def _set_web_buttons(self, enabled: bool) -> None:
        try:
            self.web_input.setEnabled(bool(enabled))
            self.web_search_button.setEnabled(bool(enabled))
            self.web_crawl_button.setEnabled(bool(enabled))
            self.web_rebuild_button.setEnabled(bool(enabled))
            self.web_results_spin.setEnabled(bool(enabled))
            self.web_auto_rebuild_checkbox.setEnabled(bool(enabled))
        except Exception:
            pass

    def _set_stop_audio_enabled(self, enabled: bool) -> None:
        self.stop_audio_button.setEnabled(enabled and self.runtime is not None)

    def _log(self, text: str) -> None:
        self.log_box.append(str(text))