"""Text chunking utilities.

Two strategies are supported:

- Character-count chunking (``chunk_text``, default) — pure text, no ML deps.
- Token-count chunking (``chunk_text_by_tokens``) — uses the tokenizer of the
  embedding model via a ``token_counter`` callable, so ``chunk_size`` matches
  what the model actually sees.

Both split on paragraph / heading boundaries and avoid cutting a chunk mid
sentence when a single block must be split.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rag_project.models import Document

SentenceSplitter = Callable[[str], list[str]]
TokenCounter = Callable[[str], int]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    token_counter: TokenCounter | None = None,
) -> list[str]:
    """Split text into chunks.

    With ``token_counter=None`` (default) chunks are measured in characters,
    preserving the original behaviour. Pass a model tokenizer to measure in
    tokens instead.
    """
    if token_counter is not None:
        return chunk_text_by_tokens(
            text,
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
            token_counter=token_counter,
        )

    _validate_chunk_params(chunk_size, chunk_overlap)

    cleaned = text.strip()
    if not cleaned:
        return []

    blocks = split_text_blocks(cleaned)
    if len(blocks) == 1 and len(blocks[0]) > chunk_size:
        return _window_chunks(blocks[0], chunk_size, chunk_overlap)

    chunks: list[str] = []
    current: list[str] = []

    for block in blocks:
        block_parts = (
            _window_chunks(block, chunk_size, chunk_overlap)
            if len(block) > chunk_size
            else [block]
        )
        for part in block_parts:
            candidate = "\n\n".join([*current, part]) if current else part
            if current and len(candidate) > chunk_size:
                previous = "\n\n".join(current).strip()
                if previous:
                    chunks.append(previous)
                current = _overlap_seed(previous, chunk_overlap)
                candidate = "\n\n".join([*current, part]) if current else part
                if len(candidate) > chunk_size:
                    current = []
                    chunks.extend(_window_chunks(part, chunk_size, chunk_overlap))
                    continue
            current.append(part)

    tail = "\n\n".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


def chunk_text_by_tokens(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    token_counter: TokenCounter,
    sentence_splitter: SentenceSplitter | None = None,
) -> list[str]:
    """Split text into chunks bounded by a token budget.

    ``token_counter`` counts tokens of a text span (e.g. the embedding model's
    ``tokenizer.encode`` length). ``sentence_splitter`` is used to split an
    oversized block at sentence boundaries instead of cutting mid-sentence.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than 0")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must not be negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    splitter = sentence_splitter or _split_sentences

    cleaned = text.strip()
    if not cleaned:
        return []

    blocks = split_text_blocks(cleaned)
    if len(blocks) == 1 and token_counter(blocks[0]) > chunk_size_tokens:
        return _window_chunks_by_tokens(
            blocks[0], chunk_size_tokens, chunk_overlap_tokens, token_counter, splitter
        )

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_counter(block)
        block_parts = (
            split_block_by_tokens(
                block, block_tokens, chunk_size_tokens, token_counter, splitter
            )
            if block_tokens > chunk_size_tokens
            else [block]
        )
        for part in block_parts:
            part_tokens = token_counter(part)
            candidate_tokens = current_tokens + part_tokens
            if current and candidate_tokens > chunk_size_tokens:
                previous = "\n\n".join(current).strip()
                if previous:
                    chunks.append(previous)
                current = _overlap_seed_tokens(
                    previous, chunk_overlap_tokens, token_counter
                )
                current_tokens = sum(token_counter(n) for n in current)
                candidate_tokens = current_tokens + part_tokens
                if candidate_tokens > chunk_size_tokens:
                    current = []
                    current_tokens = 0
                    for sub in split_block_by_tokens(
                        part,
                        part_tokens,
                        chunk_size_tokens,
                        token_counter,
                        splitter,
                    ):
                        if token_counter(sub) > chunk_size_tokens:
                            current = []
                            current_tokens = 0
                            chunks.append(sub)
                        else:
                            current = [sub]
                            current_tokens = token_counter(sub)
                    continue
            current.append(part)
            current_tokens += part_tokens

    tail = "\n\n".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


def split_text_blocks(text: str) -> list[str]:
    """Split text into paragraph and heading-aware blocks."""

    blocks: list[str] = []
    paragraph: list[str] = []

    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            _flush_paragraph(blocks, paragraph)
            continue

        if _is_heading(line):
            _flush_paragraph(blocks, paragraph)
            blocks.append(line)
            continue

        paragraph.append(line)

    _flush_paragraph(blocks, paragraph)
    return blocks or [text.strip()]


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    token_counter: TokenCounter | None = None,
) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        for index, text in enumerate(
            chunk_text(document.text, chunk_size, chunk_overlap, token_counter)
        ):
            chunks.append(
                Document(
                    id=f"{document.id}:{index:04d}",
                    text=text,
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_index": index,
                    },
                )
            )

    return chunks


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def _window_chunks_by_tokens(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    token_counter: TokenCounter,
    sentence_splitter: SentenceSplitter,
) -> list[str]:
    parts = split_block_by_tokens(
        text,
        token_counter(text),
        chunk_size_tokens,
        token_counter,
        sentence_splitter,
    )
    if not parts:
        return []
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    for part in parts:
        part_tokens = token_counter(part)
        if current_parts and current_tokens + part_tokens > chunk_size_tokens:
            previous = " ".join(current_parts).strip()
            if previous:
                chunks.append(previous)
            current_parts = _overlap_seed_tokens(
                previous, chunk_overlap_tokens, token_counter
            )
            current_tokens = sum(token_counter(n) for n in current_parts)
        if token_counter(part) > chunk_size_tokens:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_tokens = 0
            chunks.append(part)
            continue
        current_parts.append(part)
        current_tokens += part_tokens
    tail = " ".join(current_parts).strip()
    if tail:
        chunks.append(tail)
    return chunks


def split_block_by_tokens(
    block: str,
    block_tokens: int,
    chunk_size_tokens: int,
    token_counter: TokenCounter,
    sentence_splitter: SentenceSplitter,
) -> list[str]:
    """Split an oversized block into sentence-aligned pieces within the budget.

    Falls back to a token window when a single sentence itself exceeds the
    budget, so we never return an empty list for a non-empty block.
    """
    if block_tokens <= chunk_size_tokens:
        return [block]

    sentences = sentence_splitter(block)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_counter(sentence)
        if current and current_tokens + sentence_tokens > chunk_size_tokens:
            joined = " ".join(current).strip()
            if joined:
                pieces.append(joined)
            current = []
            current_tokens = 0
        if sentence_tokens > chunk_size_tokens:
            pieces.extend(_token_window(sentence, chunk_size_tokens, token_counter))
            continue
        current.append(sentence)
        current_tokens += sentence_tokens

    tail = " ".join(current).strip()
    if tail:
        pieces.append(tail)

    return pieces if pieces else [block]


def _token_window(text: str, chunk_size_tokens: int, token_counter: TokenCounter) -> list[str]:
    """Crude but safe fallback: greedily pack whole words up to the budget."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = token_counter(word)
        if current and current_tokens + word_tokens > chunk_size_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split into sentences on typical sentence-ending punctuation."""
    pieces = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [piece for piece in pieces if piece.strip()]


def _overlap_seed_tokens(
    previous: str, chunk_overlap: int, token_counter: TokenCounter
) -> list[str]:
    if chunk_overlap <= 0 or not previous:
        return []
    sentences = _split_sentences(previous)
    seed: list[str] = []
    used_tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = token_counter(sentence)
        if used_tokens + sentence_tokens > chunk_overlap:
            break
        seed.append(sentence)
        used_tokens += sentence_tokens
    seed.reverse()
    return [s for s in seed if s.strip()]


def _overlap_seed(previous: str, chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []
    seed = previous[-chunk_overlap:].strip()
    return [seed] if seed else []


def _flush_paragraph(blocks: list[str], paragraph: list[str]) -> None:
    if paragraph:
        blocks.append(" ".join(paragraph).strip())
        paragraph.clear()


def _is_heading(line: str) -> bool:
    if re.match(r"^#{1,6}\s+\S+", line):
        return True
    return len(line) <= 120 and line.endswith(":")
