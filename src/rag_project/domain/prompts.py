"""RAG answer policy: system prompt, user prompt layout and context budget."""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = (
    "Ты AI assistant для RAG-системы.\n"
    "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
    "Если информации недостаточно — так и скажи.\n"
    "Не придумывай факты.\n"
    "Если ответ найден, указывай источники ссылками вида [S1], [S2]."
)


def build_user_prompt(question: str, context_chunks: Sequence[str]) -> str:
    chunks = "\n\n".join(
        f"[{index}]\n{chunk}" for index, chunk in enumerate(context_chunks, start=1)
    )
    return (
        "Контекст:\n"
        f"{chunks}\n\n"
        "Вопрос:\n"
        f"{question}\n\n"
        "Ответь с точными ссылками на источники в формате [S1], [S2], если ответ найден."
    )


def build_messages(question: str, context_chunks: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, context_chunks)},
    ]


def limit_context_chunks(context_chunks: Sequence[str], max_chars: int) -> list[str]:
    """Trim context so its total length fits ``max_chars``, keeping chunk order."""
    if max_chars <= 0:
        return []

    limited: list[str] = []
    used = 0
    for chunk in context_chunks:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = chunk.strip()
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        limited.append(text)
        used += len(text)

    return limited
