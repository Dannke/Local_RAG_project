import unittest

from rag_project.embeddings.base import HashingEmbeddingModel
from rag_project.models import Document
from rag_project.pipelines.chat_pipeline import ChatSession
from rag_project.retrieval.retriever import Retriever
from rag_project.vectorstores.base import InMemoryVectorStore


class FakeGenerator:
    def __init__(self):
        self.context_chunks = None

    def generate(self, question, context_chunks):
        self.context_chunks = context_chunks
        return f"answer for {question}: {len(context_chunks)} chunks"

    def stream(self, question, context_chunks):
        self.context_chunks = context_chunks
        yield "stream "
        yield "answer"


class ChatPipelineTests(unittest.TestCase):
    def test_chat_session_passes_retrieved_context_to_generator(self) -> None:
        retriever = Retriever(HashingEmbeddingModel(), InMemoryVectorStore())
        retriever.index(
            [
                Document(
                    id="doc-1",
                    text="FAISS stores embeddings for semantic search.",
                    metadata={"relative_path": "notes.txt"},
                )
            ]
        )
        generator = FakeGenerator()
        session = ChatSession(retriever, generator=generator)

        response = session.ask("What stores embeddings?", top_k=1)

        self.assertIn("1 chunks", response.answer)
        self.assertIn("[S1]", generator.context_chunks[0])
        self.assertIn("FAISS stores embeddings for semantic search.", generator.context_chunks[0])
        self.assertEqual(response.contexts[0].metadata["relative_path"], "notes.txt")
        self.assertEqual(response.citations[0].label, "S1")

    def test_chat_session_streams_answer_with_retrieved_context(self) -> None:
        retriever = Retriever(HashingEmbeddingModel(), InMemoryVectorStore())
        retriever.index(
            [
                Document(
                    id="doc-1",
                    text="Streaming uses retrieved chunks.",
                    metadata={"relative_path": "notes.txt"},
                )
            ]
        )
        generator = FakeGenerator()
        session = ChatSession(retriever, generator=generator)

        response = session.stream("What streams?", top_k=1)
        answer = "".join(response.chunks)

        self.assertEqual(answer, "stream answer")
        self.assertIn("[S1]", generator.context_chunks[0])
        self.assertIn("Streaming uses retrieved chunks.", generator.context_chunks[0])
        self.assertEqual(response.results[0].document.metadata["relative_path"], "notes.txt")
