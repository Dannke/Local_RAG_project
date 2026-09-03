"""Retrieval evaluation: recall@k and MRR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402
from rag_project.config import load_settings
from rag_project.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel
from rag_project.retrieval.retriever import Retriever
from rag_project.vectorstores.faiss_store import FaissVectorStore


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_doc_ids_for_source(store: FaissVectorStore, source_path: str) -> set[int]:
    """Return FAISS IDs of documents from a given source file."""
    ids = set()
    for idx, doc in enumerate(store.documents):
        if doc.metadata.get("relative_path") == source_path:
            ids.add(store._faiss_ids[idx])
    return ids


def recall_at_k(results: list, relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = results[:k]
    found = sum(1 for r in top_k if getattr(r, "faiss_id", None) in relevant_ids)
    return found / len(relevant_ids)


def mrr(results: list, relevant_ids: set[int]) -> float:
    for rank, r in enumerate(results, start=1):
        if getattr(r, "faiss_id", None) in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(golden_path: Path, index_dir: Path, top_k: int = 5) -> dict:
    settings = load_settings()
    store = FaissVectorStore.load_from_disk(index_dir)
    retriever = Retriever(
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        vector_store=store,
    )
    golden = load_golden(golden_path)

    recall_sum = {k: 0.0 for k in [1, 3, 5, 10]}
    mrr_sum = 0.0
    total = 0

    for item in golden:
        question = item["question"]
        expected_source = item["expected_source"]
        relevant_ids = get_doc_ids_for_source(store, expected_source)

        if not relevant_ids:
            print(f"[WARN] No docs found for source: {expected_source}")
            continue

        results = retriever.search(question, top_k=top_k)
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                recall_sum[k] += recall_at_k(results, relevant_ids, k)
        mrr_sum += mrr(results, relevant_ids)
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