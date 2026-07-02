from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EmotionState:
    warmth: float = 0.75
    energy: float = 0.65
    patience: float = 0.85
    precision: float = 0.80
    playfulness: float = 0.55
    confidence: float = 0.75

    def clamp(self) -> None:
        for key, value in asdict(self).items():
            fixed = max(0.0, min(1.0, float(value)))
            setattr(self, key, fixed)

    def to_dict(self) -> dict[str, float]:
        self.clamp()
        return asdict(self)


class EmotionEngine:
    """
    Lightweight affective interaction layer.

    This does not claim the model has real emotion.
    It only controls response tone and interaction style.
    """

    def __init__(self) -> None:
        self.state = EmotionState()

    def reset(self) -> None:
        self.state = EmotionState()

    def update_from_user_text(self, text: str) -> None:
        text = str(text or "")
        lower = text.lower()

        # Debug / error situation: become more careful and patient.
        error_words = [
            "error",
            "traceback",
            "exception",
            "failed",
            "报错",
            "錯",
            "错误",
            "失败",
            "不能用",
            "坏了",
            "找不到",
        ]

        if any(word in lower or word in text for word in error_words):
            self._add("patience", 0.08)
            self._add("precision", 0.08)
            self._add("confidence", 0.03)
            self._add("warmth", 0.03)
            self._add("playfulness", -0.04)

        # Success / excitement: become more energetic and playful.
        success_words = [
            "成功",
            "牛逼",
            "快",
            "完美",
            "可以了",
            "不错",
            "不錯",
            "works",
            "great",
            "nice",
            "perfect",
            "fast",
        ]

        if any(word in lower or word in text for word in success_words):
            self._add("energy", 0.08)
            self._add("playfulness", 0.06)
            self._add("warmth", 0.04)
            self._add("confidence", 0.04)

        # Academic / thesis mode signal.
        study_words = [
            "论文",
            "thesis",
            "dissertation",
            "harvard",
            "benchmark",
            "evaluation",
            "methodology",
            "architecture",
            "rag",
        ]

        if any(word in lower or word in text for word in study_words):
            self._add("precision", 0.06)
            self._add("confidence", 0.04)
            self._add("playfulness", -0.03)

        # User wants code or direct implementation.
        code_words = [
            "代码",
            "完整代码",
            "script",
            "python",
            "powershell",
            "debug",
            "实现",
            "開搞",
            "开搞",
        ]

        if any(word in lower or word in text for word in code_words):
            self._add("precision", 0.07)
            self._add("confidence", 0.04)
            self._add("energy", 0.03)

        self.state.clamp()

    def style_prompt(self) -> str:
        s = self.state.to_dict()

        return f"""
Current emotional interaction style:
- warmth: {s["warmth"]:.2f}
- energy: {s["energy"]:.2f}
- patience: {s["patience"]:.2f}
- precision: {s["precision"]:.2f}
- playfulness: {s["playfulness"]:.2f}
- confidence: {s["confidence"]:.2f}

Use these values only to adjust tone:
- Higher warmth means more supportive wording.
- Higher precision means more exact and practical answers.
- Higher patience means calmer debugging and clearer steps.
- Higher playfulness means lightly cute phrasing.
- Higher energy means more active and encouraging tone.

Do not say these numbers unless the user asks.
Do not claim to have real feelings.
""".strip()

    def summary_text(self) -> str:
        s = self.state.to_dict()

        return (
            f"warmth={s['warmth']:.2f}, "
            f"energy={s['energy']:.2f}, "
            f"patience={s['patience']:.2f}, "
            f"precision={s['precision']:.2f}, "
            f"playfulness={s['playfulness']:.2f}, "
            f"confidence={s['confidence']:.2f}"
        )

    def to_json_data(self) -> dict[str, Any]:
        return self.state.to_dict()

    def _add(self, key: str, delta: float) -> None:
        current = float(getattr(self.state, key))
        setattr(self.state, key, current + float(delta))
        self.state.clamp()