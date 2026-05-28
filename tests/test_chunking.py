import unittest

from rag_project.ingestion.chunking import chunk_documents, chunk_text, split_text_blocks
from rag_project.models import Document


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
