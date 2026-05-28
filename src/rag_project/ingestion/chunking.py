"""Text chunking utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_project.models import Document


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
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
) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        for index, text in enumerate(chunk_text(document.text, chunk_size, chunk_overlap)):
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
