"""Unit tests for the real GDS request/response orchestration (fake transport).

Neo4j/GDS integration requires Docker, so these tests intentionally fake only
I/O; they verify procedure selection, data flow and GraphRAG compatibility.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

try:
    import neo4j  # noqa: F401
except ModuleNotFoundError:
    # Allows unit tests in lightweight CI without the Docker-image dependencies.
    stub = types.ModuleType("neo4j")
    stub.Driver = object  # type: ignore[attr-defined]
    stub.Neo4jError = RuntimeError  # type: ignore[attr-defined]
    sys.modules["neo4j"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))

from synthetic_payments import generate_synthetic_payments  # noqa: E402

from pe.axiz.graphrag_payments.application.generation import (  # noqa: E402
    DeterministicGroundedGenerator,
)
from pe.axiz.graphrag_payments.application.graphsage import (  # noqa: E402
    FEATURE_STATEMENTS,
    PROJECT,
    SIMILAR,
    TRAIN,
    WRITE,
    GraphSageNotReadyError,
    GraphSagePaymentNotFoundError,
    GraphSageService,
)
from pe.axiz.graphrag_payments.application.service import GraphRagService  # noqa: E402
from pe.axiz.graphrag_payments.domain.models import (  # noqa: E402
    ContextItem,
    GraphRagQueryRequest,
    GraphSageNeighbor,
    GraphSageSimilarResponse,
    GraphSageTrainRequest,
)
from pe.axiz.graphrag_payments.settings import Settings  # noqa: E402


class FakeGraphSage(GraphSageService):
    """Return realistic Neo4j record dicts and record each sent query."""

    def __init__(self) -> None:
        super().__init__(cast(Any, object()), Settings())
        self.calls: list[str] = []
        self.embedded = False
        self.has_payment = True

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        self.calls.append(query)
        if "RETURN gds.version()" in query:
            return [{"version": "2026.08.1"}]
        if "RETURN p LIMIT 1" in query:
            return [{"p": {"payment_id": "PAY-1007"}}] if self.has_payment else []
        if "gds.graph.exists" in query or "gds.model.exists" in query:
            return [{"exists": False}]
        if "gds.graph.project" in query:
            return [{"nodeCount": 375, "relationshipCount": 2020}]
        if "gds.beta.graphSage.train" in query:
            assert parameters["dimension"] == 32
            return [{"trainMillis": 123, "metrics": {"epochLosses": [2.4, 1.6]}}]
        if "gds.beta.graphSage.write" in query:
            self.embedded = True
            return [{"nodePropertiesWritten": 344}]
        if "MATCH (p:Payment) RETURN count(p) AS count" in query:
            return [{"count": 344}]
        if "count(p.sage_embedding)" in query:
            return [{"total": 344, "embedded": 344 if self.embedded else 0}]
        if "OPTIONAL MATCH (run:GraphSageRun" in query:
            return [{"model_name": None, "trained_at": None, "dimension": None,
                     "train_millis": None}]
        if "RETURN p.payment_id AS id" in query:
            return [{"id": "PAY-1007"}] if self.has_payment else []
        if query == SIMILAR:
            return [{"payment_id": "PAY-1003", "status": "DECLINED", "amount": 150.0,
                     "currency": "PEN", "merchant": "TravelNow", "acquirer": "Lima Payments",
                     "reason_code": "91", "similarity": 0.93}]
        return []


def test_synthetic_data_is_reproducible_bounded_and_has_unique_ids() -> None:
    rows = generate_synthetic_payments(320)
    assert rows == generate_synthetic_payments(320)
    assert len(rows) == len({row["payment_id"] for row in rows}) == 320
    assert any(row["reason_code"] == "91" for row in rows)
    assert generate_synthetic_payments(0) == []
    with pytest.raises(ValueError):
        generate_synthetic_payments(5001)


def test_train_runs_actual_graphsage_procedures_and_writes_payment_embeddings() -> None:
    service = FakeGraphSage()
    result = service.train(GraphSageTrainRequest())
    assert result.embedded_payments == 344
    assert result.epoch_losses == [2.4, 1.6]
    assert result.graph_nodes == 375
    assert service.calls.index(PROJECT) < service.calls.index(TRAIN) < service.calls.index(WRITE)
    assert all(statement in service.calls for statement in FEATURE_STATEMENTS)
    assert "nodeLabels: ['Payment']" in WRITE
    assert service.embedded


def test_train_rejects_missing_dataset() -> None:
    service = FakeGraphSage()
    service.has_payment = False
    with pytest.raises(GraphSageNotReadyError, match="Carga primero"):
        service.train(GraphSageTrainRequest())


def test_graphsage_status_and_similar_payments() -> None:
    service = FakeGraphSage()
    assert not service.status().ready
    service.embedded = True
    assert service.status().ready
    matches = service.similar("PAY-1007", top_k=3)
    assert matches.payment_id == "PAY-1007"
    assert matches.neighbors[0].similarity == 0.93
    assert SIMILAR in service.calls
    service.has_payment = False
    with pytest.raises(GraphSagePaymentNotFoundError):
        service.similar("NOT-FOUND")


def test_request_validates_graphsage_parameters() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GraphSageTrainRequest(sample_sizes=[0])
    with pytest.raises(ValidationError):
        GraphSageTrainRequest(epochs=35)
    with pytest.raises(ValidationError):
        GraphRagQueryRequest(question="short", neural_payment_id="")


class AnchoredRetriever:
    def search(self, question: str, top_k: int) -> SimpleNamespace:
        del question, top_k
        return SimpleNamespace(
            contexts=[ContextItem(
                chunk_id="KB-91", title="Código 91", text="Emisor no disponible", score=1.0,
            )],
            strategy="exact_reason_code_anchor+hybrid",
            explicit_reason_codes=["91"],
        )


def test_opt_in_neural_graphrag_preserves_exact_code_and_reports_evidence(monkeypatch: Any) -> None:
    def fake_similar(self: Any, payment_id: str, **kwargs: Any) -> GraphSageSimilarResponse:
        assert payment_id == "PAY-1007"
        return GraphSageSimilarResponse(
            payment_id=payment_id, model_name="payments_graphsage_v1",
            neighbors=[GraphSageNeighbor(
                payment_id="PAY-1003", similarity=0.93, status="DECLINED",
                amount=150, currency="PEN", acquirer="Lima Payments", reason_code="91",
            )],
        )

    monkeypatch.setattr(GraphSageService, "similar", fake_similar)
    service = GraphRagService(
        driver=cast(Any, object()), settings=Settings(),
        retriever=cast(Any, AnchoredRetriever()),
        generator=DeterministicGroundedGenerator(),
    )
    request = GraphRagQueryRequest(
        question="¿Qué pasó con el código 91?", neural_payment_id="PAY-1007", top_k=3,
    )
    response = service.query(request)
    assert response.contexts[0].chunk_id == "KB-91"
    assert response.contexts[-1].source == "graphsage"
    assert response.trace.neural_matches == 1
    assert response.neural_neighbors[0].payment_id == "PAY-1003"
    assert "0.930" in response.answer
    events = list(service.query_stream(request))
    assert events[-1]["data"]["neural_neighbors"][0]["payment_id"] == "PAY-1003"
    assert events[1]["data"]["trace"]["neural_matches"] == 1
    single = service.query(request.model_copy(update={"top_k": 1}))
    assert [context.chunk_id for context in single.contexts] == ["KB-91"]
    assert len(single.neural_neighbors) == 1
