"""Golden question set shared by the evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
