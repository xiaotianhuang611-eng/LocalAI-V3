from __future__ import annotations

from core.persona import get_persona_mode


def build_gemma_prompt(
    user_text: str,
    history_text: str = "",
    personality_mode: str = "project_engineer",
    personality_enabled: bool = True,
    emotion_prompt: str = "",
    memory_context: str = "",
    rag_context: str = "",
    live_mode: bool = False,
) -> str:
    user_text = str(user_text or "").strip()

    parts: list[str] = []

    parts.append("System:")

    if personality_enabled:
        persona = get_persona_mode(personality_mode)

        parts.append(persona.system_prompt)
        parts.append(
            "Global language rule: Always answer in English only. "
            "The user may type Chinese, but you must reply in natural English. "
            "Do not output Chinese characters. "
            "Keep the answer suitable for English XTTS voice output."
        )

    else:
        parts.append(
            "You are a fast local English voice assistant. "
            "Always answer in English only. "
            "The user may type Chinese, but you must reply in natural English. "
            "Do not output Chinese characters. "
            "Keep answers short, natural, and suitable for XTTS voice playback. "
            "Be direct, practical, and concise."
        )

    if live_mode:
        parts.append("")
        parts.append("Real-time live voice rule:")
        parts.append(
            "The assistant is in live voice mode. "
            "Answer in one or two short spoken sentences. "
            "Do not use markdown, bullet points, code blocks, tables, or long explanations. "
            "Start with the useful answer immediately. "
            "Keep each sentence short so it can be spoken quickly. "
            "If the user is practising English, give one natural correction and ask them to try again."
        )

    if personality_enabled and emotion_prompt.strip():
        parts.append("")
        parts.append("Adaptive style instruction:")
        parts.append(emotion_prompt.strip())

    if memory_context.strip():
        parts.append("")
        parts.append("Useful local memory:")
        parts.append(memory_context.strip())

    if rag_context.strip():
        parts.append("")
        parts.append("Retrieved local knowledge:")
        parts.append(rag_context.strip())

    if history_text.strip():
        parts.append("")
        parts.append("Recent conversation:")
        parts.append(history_text.strip())

    parts.append("")
    parts.append("Instruction:")

    if live_mode:
        parts.append(
            "Answer the user's latest message in English only. "
            "Use short natural spoken English. "
            "Avoid unnecessary explanation. "
            "Do not use markdown. "
            "The response should be suitable for immediate sentence-level TTS playback."
        )
    else:
        parts.append(
            "Answer the user's latest message in English only. "
            "For code or debugging, be exact and practical. "
            "For voice interaction, avoid unnecessary markdown. "
            "Keep the response reasonably short unless the user asks for full code or detailed steps."
        )

    parts.append("")
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")

    return "\n".join(parts)
