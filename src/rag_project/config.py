"""Runtime configuration for the RAG application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    vector_store_dir: Path
    embedding_model: str
    llm_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    openrouter_timeout_seconds: int
    top_k: int
    max_context_chars: int
    temperature: float
    use_reranker: bool
    reranker_model: str
    rerank_candidates: int
    chunk_size: int
    chunk_overlap: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    if env is None:
        _load_dotenv()
    source = os.environ if env is None else env
    project_root = Path(source.get("RAG_PROJECT_ROOT", _default_project_root())).resolve()
    openrouter_model = source.get(
        "OPENROUTER_MODEL",
        source.get("RAG_LLM_MODEL", "openrouter/auto"),
    )

    return Settings(
        project_root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        vector_store_dir=project_root / "data" / "vector_store",
        embedding_model=source.get(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        llm_model=source.get("RAG_LLM_MODEL", openrouter_model),
        openrouter_api_key=source.get("OPENROUTER_API_KEY"),
        openrouter_model=openrouter_model,
        openrouter_base_url=source.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_timeout_seconds=_env_int(source, "OPENROUTER_TIMEOUT_SECONDS", 60),
        top_k=_env_int(source, "TOP_K", 5),
        max_context_chars=_env_int(source, "MAX_CONTEXT_CHARS", 12_000),
        temperature=_env_float(source, "TEMPERATURE", 0.2),
        use_reranker=_env_bool(source, "USE_RERANKER", True),
        reranker_model=source.get(
            "RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
        rerank_candidates=_env_int(source, "RERANK_CANDIDATES", 20),
        chunk_size=_env_int(source, "RAG_CHUNK_SIZE", 800),
        chunk_overlap=_env_int(source, "RAG_CHUNK_OVERLAP", 120),
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(_default_project_root() / ".env", override=True)
