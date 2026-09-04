"""Health check script for the local RAG project."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.config import load_settings  # noqa: E402
from rag_project.embeddings.sentence_transformers import (  # noqa: E402
    SentenceTransformerEmbeddingModel,
)
from rag_project.llm.llm_client import OpenRouterClient  # noqa: E402
from rag_project.retrieval.retriever import Retriever  # noqa: E402
from rag_project.vectorstores.faiss_store import FaissVectorStore  # noqa: E402


def check_disk_space(path: Path, min_free_gb: float = 1.0) -> tuple[bool, str]:
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    ok = free_gb >= min_free_gb
    return ok, f"Disk free: {free_gb:.2f} GB (min {min_free_gb} GB)"


def check_index(index_dir: Path) -> tuple[bool, str]:
    try:
        store = FaissVectorStore.load_from_disk(index_dir)
        count = store.count()
        if count == 0:
            return False, f"Index loaded but empty (0 vectors) at {index_dir}"
        return True, f"Index OK: {count} vectors at {index_dir}"
    except Exception as exc:
        return False, f"Failed to load index at {index_dir}: {exc}"


def check_llm(settings) -> tuple[bool, str]:
    """Ping LLM with a minimal request."""
    try:
        client = OpenRouterClient.from_settings(
            settings,
            rate_limit_key="healthcheck",
        )
        # A tiny prompt to verify connectivity
        answer = client.generate_answer("Say OK", [])
        return True, f"LLM responded: {answer[:50]}..."
    except Exception as exc:
        return False, f"LLM check failed: {exc}"


def check_retrieval(index_dir: Path, settings) -> tuple[bool, str]:
    """Test search with a dummy query."""
    try:
        store = FaissVectorStore.load_from_disk(index_dir)
        retriever = Retriever(
            embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
            vector_store=store,
        )
        results = retriever.search("test", top_k=1)
        return True, f"Retrieval OK: {len(results)} results"
    except Exception as exc:
        return False, f"Retrieval check failed: {exc}"


def find_index_dir(default_dir: Path, chats_dir: Path) -> Path:
    """Return a usable FAISS index directory: the legacy shared path if it
    already contains an index, otherwise the most recently modified
    per-chat index directory, otherwise fall back to the legacy path
    unchanged (so the existing "not found" error message still fires).
    """
    if (default_dir / "index.faiss").exists():
        return default_dir
    if chats_dir.exists():
        candidates = sorted(
            chats_dir.glob("*/index"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for candidate in candidates:
            if (candidate / "index.faiss").exists():
                return candidate
    return default_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Health check for RAG deployment")
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Path to FAISS index directory",
        default=None,
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM connectivity check",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="Minimum free disk space in GB",
    )
    args = parser.parse_args()

    settings = load_settings()
    index_dir = args.index_dir or find_index_dir(
        settings.vector_store_dir, settings.vector_store_dir.parent / "chats"
    )

    checks = []

    # Disk space
    ok, msg = check_disk_space(settings.vector_store_dir.parent, args.min_free_gb)
    checks.append(("Disk space", ok, msg))

    # Index integrity
    ok, msg = check_index(index_dir)
    checks.append(("Index", ok, msg))

    # Retrieval
    ok, msg = check_retrieval(index_dir, settings)
    checks.append(("Retrieval", ok, msg))

    # LLM
    if not args.no_llm:
        ok, msg = check_llm(settings)
        checks.append(("LLM", ok, msg))

    all_ok = all(ok for _, ok, _ in checks)

    print("Health check results:")
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())