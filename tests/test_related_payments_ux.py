"""Regression: synthetic IDs and conversational 'relacionados' trigger both retrievals."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
from chat_flow import plan_turn  # noqa: E402
from pe.axiz.graphrag_payments.application.neural_routing import (  # noqa: E402
    payment_ids, resolve_neural_route,
)
from pe.axiz.graphrag_payments.application.graphsage import GraphSageService  # noqa: E402
from pe.axiz.graphrag_payments.application.generation import DeterministicGroundedGenerator  # noqa: E402
from pe.axiz.graphrag_payments.application.service import GraphRagService  # noqa: E402
from pe.axiz.graphrag_payments.domain.models import (  # noqa: E402
    ContextItem, GraphRagQueryRequest, GraphSageNeighbor, GraphSageSimilarResponse,
)
from pe.axiz.graphrag_payments.settings import Settings  # noqa: E402


@pytest.mark.parametrize(("text", "expected"), [
    ("dime que otros pagoso relacionado a SYN-00304 existen", ["SYN-00304"]),
    ("Busca relacionados a syn-00304", ["SYN-00304"]),
    ("Compara PAY-1008 con SYN-00304", ["PAY-1008", "SYN-00304"]),
    ("No confundas SYN-CUST-030 ni SYN-00304B con SYN-00304", ["SYN-00304"]),
])
def test_synthetic_ids_in_chat_and_backend(text: str, expected: list[str]) -> None:
    assert payment_ids(text) == expected


@pytest.mark.parametrize(("text", "previous", "action", "payment_id"), [
    ("dime que otros pagoso relacionado a SYN-00304 existen", None, "neural", "SYN-00304"),
    # Regression reported from the actual chat: the user typed "pags" instead of "pagos".
    ("dime que pags estan relacionados a SYN-00304", None, "neural", "SYN-00304"),
    ("dime qué pagos están relacionados a SYN-00304", None, "neural", "SYN-00304"),
    ("qué otras transacciones están vinculadas a SYN-00304", None, "neural", "SYN-00304"),
    ("dime que pags estan relacionados a SYN-00304", "PAY-1008", "neural", "SYN-00304"),
    ("¿Qué pagos están relacionados con SYN-00304?", None, "neural", "SYN-00304"),
    ("¿Y otros pagos relacionados?", "SYN-00304", "neural", "SYN-00304"),
    ("Busca pagos vinculados a PAY-1008", None, "neural", "PAY-1008"),
    ("¿Qué relación tiene SYN-00304 con su comercio?", None, "traditional", None),
    ("¿Qué pagos están relacionados con el código 05?", None, "traditional", None),
    ("¿Qué evidencia hay sobre fondos insuficientes y qué pagos están relacionados?", None, "traditional", None),
    ("¿Qué es el código 05?", "SYN-00304", "traditional", None),
    ("¿Qué pagos relacionados hay para SYN-00304 y PAY-1008?", None, "traditional", None),
])
def test_related_route(text: str, previous: str | None, action: str, payment_id: str | None) -> None:
    route = resolve_neural_route(text, retrieval_mode="auto", conversation_payment_id=previous)
    assert (route.action, route.payment_id) == (action, payment_id)


def test_traditional_unchanged_for_synthetic_related_payments() -> None:
    route = resolve_neural_route(
        "dime que otros pagos relacionados a SYN-00304 existen", retrieval_mode="traditional"
    )
    assert route.action == "traditional"


def test_synthetic_payment_id_persists_for_followups() -> None:
    conversation: dict[str, Any] = {"last_payment_id": None}
    plan_turn(conversation, "¿Qué ocurrió con SYN-00304?", "Neural GraphRAG inteligente")
    assert conversation["last_payment_id"] == "SYN-00304"
    followup = plan_turn(conversation, "¿Y otros pagos relacionados?", "Neural GraphRAG inteligente")
    assert followup["conversation_payment_id"] == "SYN-00304"


class Retriever:
    def search(self, question: str, top_k: int) -> SimpleNamespace:
        return SimpleNamespace(
            contexts=[ContextItem(chunk_id="KB", title="Grafo", text="Relaciones verificadas", score=.9)],
            strategy="hybrid", explicit_reason_codes=[],
        )


def test_related_payments_calls_graphsage_in_rest_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_similar(self: Any, payment_id: str, **kwargs: Any) -> GraphSageSimilarResponse:
        calls.append(payment_id)
        return GraphSageSimilarResponse(
            payment_id=payment_id, model_name="test", neighbors=[GraphSageNeighbor(
                payment_id="SYN-00305", similarity=.88, status="DECLINED", amount=10, currency="PEN"
            )],
        )

    monkeypatch.setattr(GraphSageService, "similar", fake_similar)
    service = GraphRagService(
        driver=cast(Any, object()), settings=Settings(), retriever=cast(Any, Retriever()),
        generator=DeterministicGroundedGenerator(),
    )
    request = GraphRagQueryRequest(
        question="dime que pags estan relacionados a SYN-00304", retrieval_mode="auto", top_k=4,
    )
    response = service.query(request)
    events = list(service.query_stream(request))
    assert calls == ["SYN-00304", "SYN-00304"]
    assert response.trace.neural_route == "neural"
    assert response.trace.neural_payment_id == "SYN-00304"
    assert response.contexts[0].source != "graphsage"  # GraphRAG retained
    assert response.contexts[-1].source == "graphsage"
    assert events[-1]["data"]["trace"]["neural_route"] == "neural"
    assert events[-1]["data"]["neural_neighbors"][0]["payment_id"] == "SYN-00305"


def test_sidebar_copy_shorter_and_no_required_payment_input() -> None:
    source = (ROOT / "frontend/app.py").read_text()
    assert "Con un ID: similares/relacionados → +GraphSAGE." in source
    assert "Pago actual:" in source
    assert "Referencia de esta conversación:" not in source
    assert 'key="chat_neural_payment_id"' not in source
