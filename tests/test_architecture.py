"""Enforce the Clean Architecture dependency rule: imports only point inward.

    domain  <-  application  <-  infrastructure  <-  ui / cli / bootstrap

The domain and application layers must also stay free of third-party
packages so business rules never depend on frameworks or ML libraries.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "rag_project"

# Layer -> rag_project modules it may import (prefix match).
ALLOWED_INTERNAL_IMPORTS = {
    "domain": ("rag_project.domain",),
    "application": ("rag_project.domain", "rag_project.application"),
    "infrastructure": (
        "rag_project.domain",
        "rag_project.application",
        "rag_project.infrastructure",
    ),
}
PURE_LAYERS = ("domain", "application")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _layer_modules(layer: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / layer).rglob("*.py"))


class DependencyRuleTests(unittest.TestCase):
    def test_layers_exist(self) -> None:
        for layer in ALLOWED_INTERNAL_IMPORTS:
            self.assertTrue(_layer_modules(layer), f"layer {layer!r} has no modules")

    def test_internal_imports_point_inward(self) -> None:
        for layer, allowed in ALLOWED_INTERNAL_IMPORTS.items():
            for path in _layer_modules(layer):
                for module in _imported_modules(path):
                    if not module.startswith("rag_project"):
                        continue
                    with self.subTest(layer=layer, file=path.name, imports=module):
                        self.assertTrue(
                            module.startswith(allowed),
                            f"{path.relative_to(PACKAGE_ROOT)} must not import {module}",
                        )

    def test_inner_layers_use_only_the_standard_library(self) -> None:
        for layer in PURE_LAYERS:
            for path in _layer_modules(layer):
                for module in _imported_modules(path):
                    top_level = module.split(".")[0]
                    if top_level in {"rag_project", "__future__"}:
                        continue
                    with self.subTest(layer=layer, file=path.name, imports=module):
                        self.assertTrue(
                            top_level in sys.stdlib_module_names,
                            f"{path.relative_to(PACKAGE_ROOT)} imports third-party {module}",
                        )


if __name__ == "__main__":
    unittest.main()
