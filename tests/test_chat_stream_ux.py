"""Regression tests for consistent traditional/intelligent SSE and visible GraphSAGE status."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast
from types import SimpleNamespace

import pytest

try:
    import neo4j  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("neo4j")
    stub.Driver = object  # type: ignore[attr-defined]
    exceptions = types.ModuleType("neo4j.exceptions")
    exceptions.Neo4jError = RuntimeError  # type: ignore[attr-defined]
    sys.modules["neo4j"] = stub
    sys.modules["neo4j.exceptions"] = exceptions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))

from chat_flow import plan_turn  # noqa: E402
from ui_helpers import graphsage_usage  # noqa: E402
from pe.axiz.graphrag_payments.application.generation import DeterministicGroundedGenerator  # noqa: E402
from pe.axiz.graphrag_payments.application.graphsage import GraphSageService  # noqa: E402
from pe.axiz.graphrag_payments.application.service import GraphRagService  # noqa: E402
from pe.axiz.graphrag_payments.domain.models import (  # noqa: E402
    ContextItem, GraphRagQueryRequest, GraphSageSimilarResponse, GraphSageNeighbor,
)
from pe.axiz.graphrag_payments.settings import Settings  # noqa: E402


@pytest.mark.parametrize(("mode", "expected"), [
    ("Neural GraphRAG inteligente", "auto"), ("GraphRAG tradicional", "traditional"),
])
def test_chat_turn_planning_keeps_modes_and_reference(mode: str, expected: str) -> None:
    conversation: dict[str, Any] = {"last_payment_id": None}
    first = plan_turn(conversation, "¿Qué ocurrió con PAY-1008?", mode)
    assert first["retrieval_mode"] == expected
    assert first["conversation_payment_id"] is None
    assert conversation["last_payment_id"] == "PAY-1008"
    second = plan_turn(conversation, "¿Y otros similares?", mode)
    assert second["conversation_payment_id"] == "PAY-1008"
    assert conversation["last_payment_id"] == "PAY-1008"
    plan_turn(conversation, "Compara PAY-1008 y PAY-1010", mode)
    assert conversation["last_payment_id"] is None


def test_all_graphsage_outcomes_are_disclosed() -> None:
    assert graphsage_usage({"neural_route": "neural", "neural_payment_id": "PAY-1008", "neural_matches": 3}).startswith("GraphSAGE: Sí")
    assert "PAY-1008" in graphsage_usage({"neural_route": "neural", "neural_payment_id": "PAY-1008", "neural_matches": 0})
    assert graphsage_usage({"neural_route": "traditional", "neural_note": "No solicitó similitud"}) == "GraphSAGE: No · No solicitó similitud"
    assert "GraphSAGE: No" in graphsage_usage({"neural_route": "unavailable", "neural_note": "Sin embeddings"})
    assert "GraphSAGE: No" in graphsage_usage({"neural_route": "needs_reference"})


class Retriever:
    def search(self, question: str, top_k: int) -> SimpleNamespace:
        return SimpleNamespace(contexts=[ContextItem(chunk_id="KB", title="Información", text="Dato", score=1.0)], strategy="hybrid", explicit_reason_codes=[])


def build_service() -> GraphRagService:
    return GraphRagService(
        driver=cast(Any, object()), settings=Settings(), retriever=cast(Any, Retriever()),
        generator=DeterministicGroundedGenerator(),
    )


@pytest.mark.parametrize(("mode", "question", "expected_route"), [
    ("traditional", "Busca pagos similares a PAY-1008", "traditional"),
    ("auto", "¿Qué significa el código 05?", "traditional"),
    ("auto", "Busca pagos similares a PAY-1008", "neural"),
])
def test_both_modes_stream_early_feedback_and_completion(
    monkeypatch: pytest.MonkeyPatch, mode: str, question: str, expected_route: str,
) -> None:
    calls: list[str] = []
    def fake_similar(self: Any, payment_id: str, **kwargs: Any) -> GraphSageSimilarResponse:
        calls.append(payment_id)
        return GraphSageSimilarResponse(payment_id=payment_id, model_name="test", neighbors=[GraphSageNeighbor(
            payment_id="PAY-1010", similarity=.94, status="DECLINED", amount=10.0, currency="PEN",
        )])
    monkeypatch.setattr(GraphSageService, "similar", fake_similar)
    stream = build_service().query_stream(GraphRagQueryRequest(
        question=question, retrieval_mode=mode, top_k=4,
    ))
    first = next(stream)
    assert first["event"] == "stage"
    assert ("GraphSAGE" in first["data"]["message"])  # visible before slower retrieval
    events = list(stream)
    assert any(e["event"] == "retrieval" for e in events)
    assert any(e["event"] == "delta" for e in events)
    assert events[-1]["event"] == "complete"
    trace = events[-1]["data"]["trace"]
    assert trace["neural_route"] == expected_route
    assert calls == (["PAY-1008"] if expected_route == "neural" else [])
    assert graphsage_usage(trace).startswith("GraphSAGE: Sí" if calls else "GraphSAGE: No")


def test_streaming_is_in_chat_viewport_after_composer_is_rendered() -> None:
    source = (ROOT / "frontend/app.py").read_text(encoding="utf-8")
    render = source.split("def render_chat_area(", 1)[1].split("def start_queued_turn(", 1)[0]
    assert render.index('key="chat_thread"') < render.index('assistant_slot = st.empty()')
    assert render.index('assistant_slot = st.empty()') < render.index('key="chat_composer"')
    assert render.index('key="chat_composer"') < render.index('with assistant_slot.container():')
    assert render.index('with assistant_slot.container():') < render.index('render_streaming_assistant(')
    assert 'disabled=belongs_here' in render
    # Must never make the blocking network request outside the chat viewport.
    assert source.rsplit('if submitted_question and not st.session_state.get("pending_request"):', 1)[-1].find('render_streaming_assistant(') == -1
