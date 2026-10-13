"""Application layer: use cases orchestrating the domain through ports.

Use cases depend only on the domain and on the protocols in ``ports``;
concrete adapters are injected by the composition root (``bootstrap``).
"""
