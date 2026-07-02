from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppSettings:
    voice_question_seconds: int = 30
    reference_voice_seconds: int = 8
    default_user_text: str = "What can you do for me?"
    auto_load_on_start: bool = False

    personality_mode: str = "project_engineer"
    personality_enabled: bool = True
    emotion_enabled: bool = True
    memory_enabled: bool = True
    rag_enabled: bool = False

    @classmethod
    def load(cls, root_dir: Path) -> "AppSettings":
        settings_path = Path(root_dir) / "data" / "settings.json"

        if not settings_path.exists():
            return cls()

        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))

            return cls(
                voice_question_seconds=int(data.get("voice_question_seconds", 30)),
                reference_voice_seconds=int(data.get("reference_voice_seconds", 8)),
                default_user_text=str(
                    data.get("default_user_text", "What can you do for me?")
                ),
                auto_load_on_start=bool(data.get("auto_load_on_start", False)),
                personality_mode=str(data.get("personality_mode", "project_engineer")),
                personality_enabled=bool(data.get("personality_enabled", True)),
                emotion_enabled=bool(data.get("emotion_enabled", True)),
                memory_enabled=bool(data.get("memory_enabled", True)),
                rag_enabled=bool(data.get("rag_enabled", False)),
            )

        except Exception:
            return cls()

    def save(self, root_dir: Path) -> Path:
        settings_dir = Path(root_dir) / "data"
        settings_dir.mkdir(parents=True, exist_ok=True)

        settings_path = settings_dir / "settings.json"

        settings_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return settings_path