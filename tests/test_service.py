from types import SimpleNamespace
from typing import Any, cast

from pe.axiz.graphrag_payments.application.generation import DeterministicGroundedGenerator
from pe.axiz.graphrag_payments.application.service import GraphRagService
from pe.axiz.graphrag_payments.domain.models import ContextItem, GraphRagQueryRequest
from pe.axiz.graphrag_payments.settings import Settings


class FakeRetriever:
    def search(self, question: str, top_k: int) -> SimpleNamespace:
        del question, top_k
        return SimpleNamespace(
            contexts=[
                ContextItem(
                    chunk_id="KB-1",
                    title="Control de pagos",
                    text="Contexto controlado",
                    score=0.8,
                )
            ]
        )


def test_service_returns_trace_and_context() -> None:
    settings = Settings(default_top_k=4, max_top_k=10)
    service = GraphRagService(
        driver=cast(Any, object()),
        settings=settings,
        retriever=cast(object, FakeRetriever()),  # type: ignore[arg-type]
        generator=DeterministicGroundedGenerator(),
    )

    response = service.query(GraphRagQueryRequest(question="Explica la falla de pago"))

    assert response.trace.retriever == "HybridCypherRetriever"
    assert response.trace.returned_contexts == 1
    assert response.contexts[0].chunk_id == "KB-1"
