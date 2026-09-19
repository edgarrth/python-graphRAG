from pathlib import Path
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
                    chunk_id="KB-IDEMPOTENCY",
                    title="Idempotencia para evitar cobros duplicados",
                    text="Una clave estable evita crear una nueva autorización en un retry.",
                    score=0.95,
                )
            ]
        )


def test_service_stream_emits_retrieval_deltas_and_complete() -> None:
    settings = Settings(default_top_k=4, max_top_k=10)
    service = GraphRagService(
        driver=cast(Any, object()),
        settings=settings,
        retriever=cast(Any, FakeRetriever()),
        generator=DeterministicGroundedGenerator(),
    )

    events = list(service.query_stream(GraphRagQueryRequest(question="¿Cómo evito duplicados?")))
    event_names = [event["event"] for event in events]

    assert event_names[0] == "stage"
    assert "retrieval" in event_names
    assert "delta" in event_names
    assert event_names[-1] == "complete"
    assert events[-1]["data"]["generation_provider"] == "deterministic"
    assert "Idempotencia" in events[-1]["data"]["answer"]


def test_routes_define_valid_sse_transport() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/pe/axiz/graphrag_payments/api/routes.py").read_text(encoding="utf-8")

    assert '"/api/v1/graphrag/query/stream"' in source
    assert 'media_type="text/event-stream"' in source
    assert '"X-Accel-Buffering": "no"' in source
    assert 'return f"event: {event_name}\\ndata: {payload}\\n\\n"' in source


def test_retriever_uses_linear_hybrid_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/pe/axiz/graphrag_payments/application/retrieval.py"
    ).read_text(encoding="utf-8")

    assert "effective_search_ratio=self._effective_search_ratio" in source
    assert "ranker=self._ranker" in source
    assert 'self._alpha if self._ranker == "linear" else None' in source
    assert "EXACT_REASON_CODE_QUERY" in source
    assert "exact_reason_code_anchor+hybrid" in source
