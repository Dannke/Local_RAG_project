"""Index documents from data/raw using the local prototype stack."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.pipelines.ingest_pipeline import ingest_to_faiss  # noqa: E402


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
    index_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "vector_store"
    indexed_count = ingest_to_faiss(input_dir=data_dir, index_dir=index_dir)
    print(f"Indexed {indexed_count} chunks from {data_dir} into {index_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
