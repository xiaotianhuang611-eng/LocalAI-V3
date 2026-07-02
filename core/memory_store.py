from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryData:
    version: str = "V3.3"
    assistant_name: str = "Mochi"
    project_name: str = "LocalAI_V3"

    user_preferences: dict[str, Any] = field(default_factory=dict)
    project_profile: dict[str, Any] = field(default_factory=dict)
    stable_facts: list[str] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    interaction_stats: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class MemoryStore:
    """
    Local memory store for LocalAI.

    This is not RAG yet.
    It is lightweight structured memory:
    - user preferences
    - project profile
    - stable facts
    - recent interaction summaries

    All data stays local in:
    data/memory/local_memory.json
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.memory_dir = self.root_dir / "data" / "memory"
        self.memory_path = self.memory_dir / "local_memory.json"

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.data = self._load_or_create()

    def _load_or_create(self) -> MemoryData:
        if not self.memory_path.exists():
            data = self._default_memory()
            self._save_data(data)
            return data

        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))

            data = MemoryData(
                version=str(raw.get("version", "V3.3")),
                assistant_name=str(raw.get("assistant_name", "Mochi")),
                project_name=str(raw.get("project_name", "LocalAI_V3")),
                user_preferences=dict(raw.get("user_preferences", {})),
                project_profile=dict(raw.get("project_profile", {})),
                stable_facts=list(raw.get("stable_facts", [])),
                recent_events=list(raw.get("recent_events", [])),
                interaction_stats=dict(raw.get("interaction_stats", {})),
                created_at=float(raw.get("created_at", time.time())),
                updated_at=float(raw.get("updated_at", time.time())),
            )

            self._ensure_defaults(data)
            self._save_data(data)

            return data

        except Exception:
            backup_path = self.memory_dir / f"local_memory_BROKEN_{int(time.time())}.json"

            try:
                self.memory_path.rename(backup_path)
            except Exception:
                pass

            data = self._default_memory()
            self._save_data(data)
            return data

    def _default_memory(self) -> MemoryData:
        data = MemoryData()

        data.user_preferences = {
            "language": "Chinese explanations preferred, with English technical terms when useful.",
            "coding_style": "User prefers complete code, exact file paths, and copy-pasteable commands.",
            "response_style": "Direct, practical, implementation-focused.",
            "ui_style": "Cute pastel UI style is preferred.",
            "voice_style": "Fast local voice interaction with XTTS voice cloning is preferred.",
        }

        data.project_profile = {
            "root_dir": str(self.root_dir),
            "current_version": "V3.3 Local Memory System",
            "main_goal": "Edge-native privacy-preserving multimodal STTS personal assistant.",
            "default_llm": "Gemma GGUF via llama.cpp",
            "asr": "faster-whisper small.en",
            "tts": "Coqui XTTS v2 voice cloning",
            "vision": "Qwen2.5-VL GGUF with mmproj",
            "gpu": "NVIDIA RTX 5060 Laptop GPU",
            "features": [
                "Local text chat",
                "VAD auto voice question",
                "XTTS voice cloning",
                "Qwen-VL image understanding",
                "Personality modes",
                "Emotion engine",
                "Benchmark scripts",
            ],
        }

        data.stable_facts = [
            "The project path is C:\\Users\\111\\Desktop\\LocalAI_V3.",
            "The user is building a local AI assistant, not a cloud API assistant.",
            "The assistant should prioritize speed, privacy, offline operation, and stable voice output.",
            "The user prefers full working code instead of vague guidance.",
            "The project currently uses Gemma for text, faster-whisper for ASR, XTTS for voice cloning, and Qwen-VL for image understanding.",
            "V3.1 added VAD automatic silence detection.",
            "V3.2 added Personality Mode and Emotion Engine.",
            "V3.3 adds Local Memory System.",
        ]

        data.interaction_stats = {
            "total_turns": 0,
            "last_personality_mode": "project_engineer",
            "last_emotion_summary": "",
        }

        return data

    def _ensure_defaults(self, data: MemoryData) -> None:
        default_data = self._default_memory()

        for key, value in default_data.user_preferences.items():
            data.user_preferences.setdefault(key, value)

        for key, value in default_data.project_profile.items():
            data.project_profile.setdefault(key, value)

        existing = set(data.stable_facts)

        for fact in default_data.stable_facts:
            if fact not in existing:
                data.stable_facts.append(fact)

        data.interaction_stats.setdefault("total_turns", 0)
        data.interaction_stats.setdefault("last_personality_mode", "project_engineer")
        data.interaction_stats.setdefault("last_emotion_summary", "")

    def save(self) -> None:
        self.data.updated_at = time.time()
        self._save_data(self.data)

    def _save_data(self, data: MemoryData) -> None:
        data.updated_at = time.time()

        self.memory_path.write_text(
            json.dumps(asdict(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve_context(self, query: str, max_chars: int = 1800) -> str:
        query = str(query or "").lower()

        sections: list[str] = []

        sections.append("User preferences:")
        for key, value in self.data.user_preferences.items():
            sections.append(f"- {key}: {value}")

        sections.append("")
        sections.append("Project profile:")
        profile = self.data.project_profile

        for key, value in profile.items():
            if isinstance(value, list):
                sections.append(f"- {key}: {', '.join(str(x) for x in value)}")
            else:
                sections.append(f"- {key}: {value}")

        matched_facts = self._match_facts(query)

        if matched_facts:
            sections.append("")
            sections.append("Relevant stable facts:")
            for fact in matched_facts:
                sections.append(f"- {fact}")
        else:
            sections.append("")
            sections.append("Stable facts:")
            for fact in self.data.stable_facts[:8]:
                sections.append(f"- {fact}")

        recent = self.data.recent_events[-5:]

        if recent:
            sections.append("")
            sections.append("Recent local events:")
            for event in recent:
                label = event.get("label", "event")
                content = event.get("content", "")
                sections.append(f"- {label}: {content}")

        text = "\n".join(sections).strip()

        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0].strip()

        return text

    def _match_facts(self, query: str) -> list[str]:
        if not query:
            return self.data.stable_facts[:8]

        keywords = [
            "path",
            "路径",
            "localai",
            "项目",
            "project",
            "gemma",
            "xtts",
            "voice",
            "语音",
            "qwen",
            "vision",
            "image",
            "图片",
            "vad",
            "memory",
            "rag",
            "benchmark",
            "论文",
            "thesis",
            "dissertation",
            "privacy",
            "隐私",
        ]

        active_keywords = [word for word in keywords if word.lower() in query]

        if not active_keywords:
            return self.data.stable_facts[:6]

        matched: list[str] = []

        for fact in self.data.stable_facts:
            lower_fact = fact.lower()

            if any(word.lower() in lower_fact for word in active_keywords):
                matched.append(fact)

        return matched[:10]

    def add_stable_fact(self, fact: str) -> None:
        fact = str(fact or "").strip()

        if not fact:
            return

        if fact not in self.data.stable_facts:
            self.data.stable_facts.append(fact)
            self.save()

    def set_preference(self, key: str, value: Any) -> None:
        key = str(key or "").strip()

        if not key:
            return

        self.data.user_preferences[key] = value
        self.save()

    def set_project_value(self, key: str, value: Any) -> None:
        key = str(key or "").strip()

        if not key:
            return

        self.data.project_profile[key] = value
        self.save()

    def add_event(self, label: str, content: str, max_events: int = 30) -> None:
        label = str(label or "event").strip()
        content = str(content or "").strip()

        if not content:
            return

        event = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "content": content[:500],
        }

        self.data.recent_events.append(event)

        if len(self.data.recent_events) > max_events:
            self.data.recent_events = self.data.recent_events[-max_events:]

        self.save()

    def update_from_interaction(
        self,
        user_text: str,
        assistant_text: str,
        personality_mode: str,
        emotion_summary: str,
    ) -> None:
        user_text = str(user_text or "").strip()
        assistant_text = str(assistant_text or "").strip()

        self.data.interaction_stats["total_turns"] = (
            int(self.data.interaction_stats.get("total_turns", 0)) + 1
        )
        self.data.interaction_stats["last_personality_mode"] = str(personality_mode)
        self.data.interaction_stats["last_emotion_summary"] = str(emotion_summary)

        summary = self._make_interaction_summary(user_text, assistant_text)
        self.add_event("conversation", summary)

        self._auto_learn_lightweight(user_text)

        self.save()

    def _make_interaction_summary(self, user_text: str, assistant_text: str) -> str:
        user_preview = user_text.replace("\n", " ")[:160]
        assistant_preview = assistant_text.replace("\n", " ")[:160]

        return f"User asked: {user_preview} | Assistant replied: {assistant_preview}"

    def _auto_learn_lightweight(self, user_text: str) -> None:
        text = str(user_text or "")
        lower = text.lower()

        if "完整代码" in text or "完整代碼" in text:
            self.set_preference(
                "code_delivery",
                "User strongly prefers complete full-file code when modifying the project.",
            )

        if "不要" in text and "中英" in text:
            self.set_preference(
                "language_mode",
                "Do not add unnecessary Chinese-English mode unless explicitly requested.",
            )

        if "benchmark" in lower:
            self.add_stable_fact(
                "The user wants rich benchmark evaluation including latency, RTF, memory, stability, and charts."
            )

        if "rag" in lower:
            self.add_stable_fact(
                "The project is planned to add local RAG after the local memory system."
            )

    def short_summary(self) -> str:
        stats = self.data.interaction_stats

        return (
            f"Memory V{self.data.version} | "
            f"facts={len(self.data.stable_facts)} | "
            f"events={len(self.data.recent_events)} | "
            f"turns={stats.get('total_turns', 0)} | "
            f"mode={stats.get('last_personality_mode', 'unknown')}"
        )

    def export_text(self) -> str:
        return self.retrieve_context(query="", max_chars=5000)

    def clear_recent_events(self) -> None:
        self.data.recent_events.clear()
        self.save()

    def reset_all(self) -> None:
        self.data = self._default_memory()
        self.save()