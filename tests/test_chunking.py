import unittest

from rag_project.ingestion.chunking import (
    chunk_documents,
    chunk_text,
    chunk_text_by_tokens,
    split_text_blocks,
)
from rag_project.models import Document


def _word_token_counter(text: str) -> int:
    return len(text.split())


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_respects_overlap(self) -> None:
        chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)

        self.assertEqual(chunks, ["abcd", "defg", "ghij"])

    def test_chunk_documents_keeps_parent_metadata(self) -> None:
        document = Document(id="doc-1", text="abcdefghij", metadata={"source": "example.txt"})

        chunks = chunk_documents([document], chunk_size=5, chunk_overlap=0)

        self.assertEqual([chunk.id for chunk in chunks], ["doc-1:0000", "doc-1:0001"])
        self.assertEqual(chunks[0].metadata["parent_id"], "doc-1")
        self.assertEqual(chunks[0].metadata["source"], "example.txt")

    def test_split_text_blocks_respects_headings_and_paragraphs(self) -> None:
        text = "# Heading\n\nFirst paragraph\ncontinues here.\n\nNext:"

        blocks = split_text_blocks(text)

        self.assertEqual(blocks, ["# Heading", "First paragraph continues here.", "Next:"])

    def test_chunk_text_prefers_paragraph_boundaries(self) -> None:
        text = "Intro paragraph.\n\nSecond paragraph with details."

        chunks = chunk_text(text, chunk_size=32, chunk_overlap=0)

        self.assertEqual(chunks, ["Intro paragraph.", "Second paragraph with details."])


class TokenChunkingTests(unittest.TestCase):
    def test_short_text_is_single_chunk(self) -> None:
        text = "One two three four five six seven eight."
        chunks = chunk_text_by_tokens(
            text, chunk_size_tokens=20, chunk_overlap_tokens=2, token_counter=_word_token_counter
        )
        self.assertEqual(chunks, [text])

    def test_splits_long_text_by_token_budget(self) -> None:
        words = " ".join(f"token{i:02d}" for i in range(10))
        chunks = chunk_text_by_tokens(
            words, chunk_size_tokens=4, chunk_overlap_tokens=0, token_counter=_word_token_counter
        )
        for chunk in chunks:
            self.assertLessEqual(_word_token_counter(chunk), 4)
        self.assertGreaterEqual(len(chunks), 3)

    def test_does_not_cut_mid_sentence(self) -> None:
        sentences = [
            "First sentence with several words here.",
            "Second sentence with a few more words.",
            "Third sentence that is longer still okay.",
            "Fourth sentence finishing the text off.",
        ]
        text = " ".join(sentences)
        # Budget large enough that each sentence fits whole, so chunks should
        # only ever break on sentence boundaries.
        budget = max(_word_token_counter(s) for s in sentences) + 1
        chunks = chunk_text_by_tokens(
            text,
            chunk_size_tokens=budget,
            chunk_overlap_tokens=0,
            token_counter=_word_token_counter,
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            normalized = chunk.strip()
            self.assertTrue(
                normalized.endswith((".", "!", "?")),
                msg=f"Sentence cut mid-way: {normalized!r}",
            )

    def test_documents_use_token_counter_when_provided(self) -> None:
        document = Document(id="doc", text="one two three four five six seven eight nine ten")
        chunks = chunk_documents(
            [document],
            chunk_size=3,
            chunk_overlap=0,
            token_counter=_word_token_counter,
        )
        for chunk in chunks:
            self.assertLessEqual(_word_token_counter(chunk.text), 3)

    def test_char_chunking_unchanged_without_counter(self) -> None:
        chunks = chunk_documents(
            [Document(id="d", text="abcdefghij")], chunk_size=5, chunk_overlap=0
        )
        self.assertEqual([c.text for c in chunks], ["abcde", "fghij"])
