import unittest

from rag_project.models import Document, SearchResult
from rag_project.retrieval.citations import build_citations, format_citation_context
from rag_project.retrieval.reranker import NoOpReranker


class RerankerAndCitationTests(unittest.TestCase):
    def test_noop_reranker_keeps_existing_order(self) -> None:
        results = [
            SearchResult(Document(id="a", text="A"), score=0.1),
            SearchResult(Document(id="b", text="B"), score=0.9),
        ]

        reranked = NoOpReranker().rerank("query", results, top_k=1)

        self.assertEqual(reranked[0].document.id, "a")

    def test_citations_include_source_page_chunk_and_label(self) -> None:
        results = [
            SearchResult(
                Document(
                    id="doc:0001",
                    text="Evidence text.",
                    metadata={
                        "relative_path": "report.pdf",
                        "page": 2,
                        "chunk_index": 3,
                    },
                ),
                score=0.75,
            )
        ]

        citations = build_citations(results)
        chunks = format_citation_context(citations)

        self.assertEqual(citations[0].label, "S1")
        self.assertEqual(citations[0].source, "report.pdf")
        self.assertEqual(citations[0].page, "2")
        self.assertIn("[S1]", chunks[0])
        self.assertIn("page=2", chunks[0])
