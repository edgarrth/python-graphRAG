"""Regression checks: default chat payload, opt-in GraphSAGE and Neo4j 6 import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))

from api_client import ApiClient  # noqa: E402


@pytest.mark.parametrize(
    ("payment_id", "expected"),
    [
        (None, {"question": "¿Qué ocurrió con PAY-1007?", "top_k": 4, "include_context": True}),
        ("PAY-1007", {
            "question": "¿Qué ocurrió con PAY-1007?", "top_k": 4,
            "include_context": True, "neural_payment_id": "PAY-1007",
        }),
    ],
)
def test_http_chat_payload_keeps_original_fields_and_opt_in_neural_id(
    monkeypatch: pytest.MonkeyPatch, payment_id: str | None, expected: dict[str, Any]
) -> None:
    actual: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        actual.update({"url": url, **kwargs})
        return httpx.Response(200, json={"answer": "OK"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = ApiClient().query("¿Qué ocurrió con PAY-1007?", 4, neural_payment_id=payment_id)
    assert result["answer"] == "OK"
    assert actual["url"].endswith("/api/v1/graphrag/query")
    assert actual["json"] == expected


@pytest.mark.parametrize("payment_id", [None, "PAY-1007"])
def test_sse_chat_payload_is_backward_compatible(
    monkeypatch: pytest.MonkeyPatch, payment_id: str | None
) -> None:
    actual: dict[str, Any] = {}

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> Any:
            return iter(["event: complete", 'data: {"answer": "OK"}', ""])

    def fake_stream(method: str, url: str, **kwargs: Any) -> FakeStream:
        actual.update({"method": method, "url": url, **kwargs})
        return FakeStream()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    events = list(ApiClient().query_stream("¿Qué ocurrió con PAY-1007?", 4, payment_id))
    assert actual["method"] == "POST"
    assert actual["url"].endswith("/api/v1/graphrag/query/stream")
    assert actual["json"]["question"] == "¿Qué ocurrió con PAY-1007?"
    assert actual["json"].get("neural_payment_id") == payment_id
    assert ("neural_payment_id" in actual["json"]) == (payment_id is not None)
    assert events == [{"event": "complete", "data": {"answer": "OK"}}]


def test_graphsage_import_works_without_neo4j_error_at_driver_package_root() -> None:
    """Reproduce 6.x layout: exceptions.Neo4jError exists, neo4j.Neo4jError doesn't."""
    snippet = """
import sys, types
neo4j = types.ModuleType('neo4j')
neo4j.__path__ = []
neo4j.Driver = object
exceptions = types.ModuleType('neo4j.exceptions')
exceptions.Neo4jError = type('Neo4jError', (Exception,), {})
sys.modules['neo4j'] = neo4j
sys.modules['neo4j.exceptions'] = exceptions
from pe.axiz.graphrag_payments.application.graphsage import GraphSageService
assert GraphSageService.__name__ == 'GraphSageService'
assert not hasattr(neo4j, 'Neo4jError')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    run = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, check=False, env=env, timeout=15,
    )
    assert run.returncode == 0, run.stderr


def test_chat_ui_wires_automatic_routing_and_preserves_traditional_mode() -> None:
    source = (ROOT / "frontend" / "app.py").read_text(encoding="utf-8")
    assert '"retrieval_mode": "Neural GraphRAG inteligente"' in source
    assert 'options=["Neural GraphRAG inteligente", "GraphRAG tradicional"]' in source
    assert 'key="chat_neural_payment_id"' not in source
    assert 'conversation_payment_id=st.session_state.pending_conversation_payment_id' in source
    assert 'conversation["last_payment_id"] = None' in source
    assert 'st.dataframe(payload["neural_neighbors"]' in source
    assert 'client.query_stream(' in source
    docker = (ROOT / "infrastructure" / "frontend.Dockerfile").read_text()
    assert "COPY src/pe ./pe" in docker


def test_conversational_client_sends_mode_and_previous_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"answer": "OK"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    ApiClient().query("¿Y otros similares?", 4, retrieval_mode="auto", conversation_payment_id="PAY-1008")
    assert captured["json"] == {
        "question": "¿Y otros similares?", "top_k": 4, "include_context": True,
        "retrieval_mode": "auto", "conversation_payment_id": "PAY-1008",
    }
    ApiClient().query("¿Qué es el código 05?", 4)
    assert captured["json"] == {
        "question": "¿Qué es el código 05?", "top_k": 4, "include_context": True,
    }
