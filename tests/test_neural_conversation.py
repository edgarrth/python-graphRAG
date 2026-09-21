"""Validate conversational routing independently of LLM and external Neo4j services."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from pe.axiz.graphrag_payments.application.generation import DeterministicGroundedGenerator
from pe.axiz.graphrag_payments.application.graphsage import (
    GraphSageNotReadyError, GraphSageService,
)
from pe.axiz.graphrag_payments.application.neural_routing import (
    payment_ids, resolve_neural_route,
)
from pe.axiz.graphrag_payments.application.service import GraphRagService
from pe.axiz.graphrag_payments.domain.models import (
    ContextItem, GraphRagQueryRequest, GraphSageNeighbor, GraphSageSimilarResponse,
)
from pe.axiz.graphrag_payments.settings import Settings


class Retriever:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def search(self, question: str, top_k: int) -> SimpleNamespace:
        self.questions.append(question)
        return SimpleNamespace(contexts=[ContextItem(
            chunk_id="KB-05", title="Rechazos", text="Contexto confirmado", score=0.9,
        )], strategy="hybrid", explicit_reason_codes=[])


def build_service(retriever: Retriever) -> GraphRagService:
    return GraphRagService(
        driver=cast(Any, object()), settings=Settings(), retriever=cast(Any, retriever),
        generator=DeterministicGroundedGenerator(),
    )


@pytest.mark.parametrize(("question", "expected"), [
    ("Revisa PAY-1008, por favor.", ["PAY-1008"]),
    ("Compara pay-1008 y PAY-1010; no confundir con PAY-1008", ["PAY-1008", "PAY-1010"]),
    ("El código 05 causó 123 rechazos y PAY-1008A no es ID válido", []),
    ("Mira PAY-999 (código 91)", ["PAY-999"]),
])
def test_extracts_only_explicit_payment_ids(question: str, expected: list[str]) -> None:
    assert payment_ids(question) == expected


@pytest.mark.parametrize(("question", "mode", "context", "expected_action", "expected_id"), [
    ("¿Qué es la idempotencia?", "auto", None, "traditional", None),
    ("¿Qué pasó con PAY-1008?", "auto", None, "traditional", None),
    ("Busca pagos similares a PAY-1008", "traditional", None, "traditional", None),
    ("Busca pagos similares a pay-1008", "auto", None, "neural", "PAY-1008"),
    ("¿Y otros parecidos?", "auto", "PAY-1008", "neural", "PAY-1008"),
    ("¿Y otros similares?", "auto", None, "needs_reference", None),
    ("Compara PAY-1008 y PAY-1010", "auto", "PAY-9999", "traditional", None),
    ("Compara los códigos 05 y 91", "auto", "PAY-1008", "traditional", None),
    ("Busca similares a PAY-1008 y PAY-1010", "auto", None, "multiple_references", None),
    ("¿Y otros similares?", "auto", "NO-ES-PAGO", "needs_reference", None),
    ("¿Qué significa el código 05?", "auto", "PAY-1008", "traditional", None),
])
def test_routes_neural_only_when_reference_is_clear(
    question: str, mode: str, context: str | None, expected_action: str, expected_id: str | None,
) -> None:
    route = resolve_neural_route(question, retrieval_mode=mode, conversation_payment_id=context)
    assert route.action == expected_action
    assert route.payment_id == expected_id


def test_legacy_explicit_neural_id_is_still_honored() -> None:
    route = resolve_neural_route("¿Qué pasó con código 05?", neural_payment_id="PAY-1007")
    assert route.action == "neural" and route.payment_id == "PAY-1007"


def test_general_question_does_not_invoke_graphsage(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_similar(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No debería ejecutarse GraphSAGE")

    monkeypatch.setattr(GraphSageService, "similar", no_similar)
    retriever = Retriever()
    response = build_service(retriever).query(GraphRagQueryRequest(
        question="¿Por qué se rechaza un pago con código 05?", retrieval_mode="auto",
        conversation_payment_id="PAY-1008",
    ))
    assert response.trace.neural_route == "traditional"
    assert response.trace.neural_matches == 0
    assert response.contexts[0].chunk_id == "KB-05"
    assert len(retriever.questions) == 1


def test_followup_resolves_previous_reference_in_rest_and_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def fake_similar(self: Any, payment_id: str, **kwargs: Any) -> GraphSageSimilarResponse:
        requested.append(payment_id)
        return GraphSageSimilarResponse(
            payment_id=payment_id, model_name="test",
            neighbors=[GraphSageNeighbor(
                payment_id="PAY-1009", similarity=0.91, status="DECLINED",
                amount=10, currency="PEN",
            )],
        )

    monkeypatch.setattr(GraphSageService, "similar", fake_similar)
    retriever = Retriever()
    service = build_service(retriever)
    request = GraphRagQueryRequest(
        question="¿Y otros pagos parecidos?", retrieval_mode="auto",
        conversation_payment_id="PAY-1008", top_k=3,
    )
    result = service.query(request)
    events = list(service.query_stream(request))
    assert requested == ["PAY-1008", "PAY-1008"]
    assert "pago de referencia: PAY-1008" in retriever.questions[0]
    assert result.trace.neural_payment_id == "PAY-1008"
    assert result.trace.neural_route == "neural"
    assert result.contexts[-1].source == "graphsage"
    assert events[-1]["data"]["trace"]["neural_payment_id"] == "PAY-1008"
    assert events[-1]["data"]["neural_neighbors"][0]["payment_id"] == "PAY-1009"


@pytest.mark.parametrize("question", [
    "Encuentra pagos similares", "Busca similares a PAY-1008 y PAY-1010",
])
def test_ambiguous_reference_asks_for_clarification_without_search(question: str) -> None:
    retriever = Retriever()
    service = build_service(retriever)
    req = GraphRagQueryRequest(question=question, retrieval_mode="auto")
    response = service.query(req)
    events = list(service.query_stream(req))
    assert not retriever.questions
    assert "pago" in response.answer.lower()
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["answer"] == response.answer
    assert events[-1]["data"]["trace"]["neural_matches"] == 0


def test_untrained_graphsage_does_not_break_normal_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(self: Any, payment_id: str, **kwargs: Any) -> Any:
        raise GraphSageNotReadyError("Embeddings no disponibles")

    monkeypatch.setattr(GraphSageService, "similar", unavailable)
    response = build_service(Retriever()).query(GraphRagQueryRequest(
        question="Busca pagos similares a PAY-1008", retrieval_mode="auto",
    ))
    assert response.trace.neural_route == "unavailable"
    assert response.trace.neural_matches == 0
    assert response.trace.neural_note and "no disponibles" in response.trace.neural_note
    assert response.contexts[0].chunk_id == "KB-05"
