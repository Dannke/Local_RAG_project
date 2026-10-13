"""Retrieval evaluation: recall@k and MRR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared eval helpers

# ruff: noqa: E402
from golden import load_golden

from rag_project.bootstrap import create_retriever
from rag_project.config import load_settings
from rag_project.domain.models import Document, SearchResult
from rag_project.infrastructure.faiss_store import FaissVectorStore


def _from_source(document: Document, source_path: str) -> bool:
    return document.relative_path == source_path


def count_chunks_for_source(documents: list[Document], source_path: str) -> int:
    """Number of indexed chunks that belong to a given source file."""
    return sum(1 for document in documents if _from_source(document, source_path))


def recall_at_k(
    results: list[SearchResult], source_path: str, relevant_count: int, k: int
) -> float:
    if not relevant_count:
        return 1.0
    found = sum(1 for result in results[:k] if _from_source(result.document, source_path))
    return found / relevant_count


def mrr(results: list[SearchResult], source_path: str) -> float:
    for rank, result in enumerate(results, start=1):
        if _from_source(result.document, source_path):
            return 1.0 / rank
    return 0.0


def evaluate(golden_path: Path, index_dir: Path, top_k: int = 5) -> dict:
    retriever = create_retriever(index_dir, load_settings())
    store = retriever.vector_store
    assert isinstance(store, FaissVectorStore)
    documents = store.documents
    golden = load_golden(golden_path)

    recall_sum = {k: 0.0 for k in [1, 3, 5, 10]}
    mrr_sum = 0.0
    total = 0

    for item in golden:
        question = item["question"]
        expected_source = item["expected_source"]
        relevant_count = count_chunks_for_source(documents, expected_source)

        if not relevant_count:
            print(f"[WARN] No docs found for source: {expected_source}")
            continue

        results = retriever.search(question, top_k=top_k)
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                recall_sum[k] += recall_at_k(results, expected_source, relevant_count, k)
        mrr_sum += mrr(results, expected_source)
        total += 1

    return {
        "recall@1": recall_sum[1] / total if total else 0,
        "recall@3": recall_sum[3] / total if total else 0,
        "recall@5": recall_sum[5] / total if total else 0,
        "recall@10": recall_sum[10] / total if total else 0,
        "mrr": mrr_sum / total if total else 0,
        "questions_evaluated": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_set.jsonl")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "retrieval_metrics.json")
    args = parser.parse_args()

    settings = load_settings()
    index_dir = args.index_dir or settings.vector_store_dir

    metrics = evaluate(args.golden, index_dir, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
