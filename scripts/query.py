"""Ask a question over documents in data/raw using the local prototype stack."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.pipelines.search_pipeline import search_index  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "Your question" [data_dir]')
        return 2

    question = sys.argv[1]
    index_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "vector_store"

    results = search_index(question, index_dir=index_dir)
    for number, result in enumerate(results, start=1):
        source = result.document.metadata.get("relative_path", result.document.id)
        print(f"[{number}] score={result.score:.4f} source={source}")
        print(result.document.text[:700].strip())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
