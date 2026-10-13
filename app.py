"""Streamlit entry point for the local RAG project: ``streamlit run app.py``."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Suppress optional dependency warnings from transformers BEFORE any imports.
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", module=".*transformers.*")

BASE_DIR = Path(__file__).parent
# Make the package importable without `pip install -e .` during development.
sys.path.insert(0, str(BASE_DIR / "src"))

from rag_project.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main(style_path=BASE_DIR / "assets" / "styles.css")
