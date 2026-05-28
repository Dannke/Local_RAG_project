import unittest
from pathlib import Path

from rag_project.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_load_settings_from_env(self) -> None:
        settings = load_settings(
            {
                "RAG_PROJECT_ROOT": ".",
                "RAG_EMBEDDING_MODEL": "test-embeddings",
            "RAG_LLM_MODEL": "test-llm",
            "RAG_CHUNK_SIZE": "200",
            "RAG_CHUNK_OVERLAP": "20",
            "TOP_K": "3",
            "MAX_CONTEXT_CHARS": "1000",
            "TEMPERATURE": "0.4",
            "USE_RERANKER": "false",
            "RERANKER_MODEL": "test-reranker",
            "RERANK_CANDIDATES": "9",
        }
    )

        self.assertEqual(settings.project_root, Path(".").resolve())
        self.assertEqual(settings.embedding_model, "test-embeddings")
        self.assertEqual(settings.llm_model, "test-llm")
        self.assertEqual(settings.chunk_size, 200)
        self.assertEqual(settings.chunk_overlap, 20)
        self.assertEqual(settings.top_k, 3)
        self.assertEqual(settings.max_context_chars, 1000)
        self.assertEqual(settings.temperature, 0.4)
        self.assertFalse(settings.use_reranker)
        self.assertEqual(settings.reranker_model, "test-reranker")
        self.assertEqual(settings.rerank_candidates, 9)
