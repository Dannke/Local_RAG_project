"""Domain layer: entities and pure business rules of the RAG system.

Nothing here performs I/O or imports third-party libraries, so every rule
(chunking, citations, prompt policy, manifest diffing) is testable in isolation.
"""
