"""Regression: the API must import Neo4jError from the driver exceptions module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_api_routes_import_neo4j_error_from_exceptions() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/pe/axiz/graphrag_payments/api/routes.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(
        imp.module == "neo4j.exceptions"
        and any(alias.name == "Neo4jError" for alias in imp.names)
        for imp in imports
    )
    assert not any(
        imp.module == "neo4j"
        and any(alias.name == "Neo4jError" for alias in imp.names)
        for imp in imports
    )
