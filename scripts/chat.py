"""Start an interactive RAG chat session over the saved FAISS index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.bootstrap import create_chat_session  # noqa: E402
from rag_project.cli import run_chat_loop  # noqa: E402


def main() -> int:
    index_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "vector_store"
    return run_chat_loop(create_chat_session(index_dir=index_dir))


if __name__ == "__main__":
    raise SystemExit(main())
