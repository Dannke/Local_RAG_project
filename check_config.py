#!/usr/bin/env python
"""Проверить, что Settings имеет необходимые атрибуты."""

from rag_project.config import load_settings

settings = load_settings()

print(f"✓ Settings loaded successfully")
print(f"  - use_reranker: {getattr(settings, 'use_reranker', 'MISSING')}")
print(f"  - reranker_model: {getattr(settings, 'reranker_model', 'MISSING')}")
print(f"  - rerank_candidates: {getattr(settings, 'rerank_candidates', 'MISSING')}")

# Проверим все атрибуты
print(f"\nВсе атрибуты Settings:")
for attr in dir(settings):
    if not attr.startswith('_'):
        value = getattr(settings, attr)
        if not callable(value):
            print(f"  - {attr}: {type(value).__name__}")
