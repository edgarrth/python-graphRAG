"""Regression tests for real-embedding orchestration and honest offline evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from benchmark_case import generate_benchmark_payments  # noqa: E402

from pe.axiz.graphrag_payments.application.benchmark import (  # noqa: E402
    COHORT_CYPHER,
    NEURAL_CYPHER,
    BenchmarkNotReadyError,
    BenchmarkService,
    business_score,
    load_truth,
    ranking_metrics,
)


def test_cohort_is_reproducible_truth_is_separate_and_hard_negatives_exist() -> None:
    rows = generate_benchmark_payments()
    truth = load_truth()
    assert len(rows) == len({p["payment_id"] for p in rows}) == 48
    assert rows == generate_benchmark_payments()
    assert len(truth["anchors"]) == 8
    assert len(truth["hard_negatives"]) == 16
    assert len({p for group in truth["incident_members"].values() for p in group}) == 32
    assert all("incident_id" not in row and "relevant" not in row for row in rows)
    assert not any("incident" in col or "truth" in col for col in COHORT_CYPHER.split("RETURN")[1].split("ORDER BY")[0].split(","))
    assert "vector.similarity.cosine" in NEURAL_CYPHER
    assert "sage_embedding" in NEURAL_CYPHER


def test_metrics_use_true_fixed_budget_and_binary_ndcg() -> None:
    metrics = ranking_metrics(["R1", "N1", "R2"], {"R1", "R2", "R3"}, 3)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.hit_at_k == 1
    assert 0 < metrics.ndcg_at_k < 1
    assert ranking_metrics([], {"R"}, 5).precision_at_k == 0


def test_business_rules_never_use_hidden_truth_and_reject_none_joins() -> None:
    row = {"merchant_id": None, "acquirer_id": "A", "reason_code": "91", "customer_id": None,
           "method_id": None, "created_at": "2026-09-18T10:00:00-05:00", "status": "FAILED", "amount": 100}
    other = {**row, "created_at": "2026-09-18T10:06:00-05:00", "amount": 105}
    score, reasons = business_score(row, other)
    assert score == 11
    assert "mismo cliente" not in reasons and "mismo método de pago" not in reasons
    other["acquirer_id"] = "B"
    assert business_score(row, other)[0] == score - 4


class FakeDriver:
    def __init__(self, *, has_vectors: bool = True, incomplete: bool = False) -> None:
        self.calls: list[str] = []
        self.has_vectors, self.incomplete = has_vectors, incomplete
        self.rows = generate_benchmark_payments()

    def execute_query(self, statement: str, **params: Any) -> tuple[list[dict[str, Any]], None, None]:
        self.calls.append(statement)
        if statement == COHORT_CYPHER:
            rows = self.rows[1:] if self.incomplete else self.rows
            return ([{
                "payment_id": row["payment_id"], "amount": row["amount"],
                "status": row["status"], "created_at": row["created_at"],
                "merchant_id": row["merchant_id"], "acquirer_id": row["acquirer_id"],
                "reason_code": row["reason_code"], "customer_id": row["customer_id"],
                "method_id": row["method_id"], "embedded": self.has_vectors,
            } for row in rows], None, None)
        assert statement == NEURAL_CYPHER
        truth = load_truth()
        # Only fake the transport of returned cosine scores. The application
        # never uses truth to rank in this fake; it is used below for assertions.
        return ([{"anchor": a, "candidate": p["payment_id"],
                  "score": (48 - i) / 100} for a in params["anchors"]
                 for i, p in enumerate(self.rows) if p["payment_id"] != a], None, None)


def test_full_benchmark_compares_same_queries_and_reports_novel_hits() -> None:
    fake = FakeDriver()
    report = BenchmarkService(fake, "neo4j").run(top_k=5)
    assert report.synthetic and report.cohort_payments == 48 and report.queries == 8
    assert fake.calls == [COHORT_CYPHER, NEURAL_CYPHER]
    assert len(report.results) == 8
    for query in report.results:
        assert len(query.baseline) == len(query.graphsage) == 5
        assert set(query.additional_relevant_ids) <= {
            p.payment_id for p in query.graphsage if p.relevant_in_synthetic_truth
        }
        assert not (set(query.additional_relevant_ids) & {p.payment_id for p in query.baseline})
    assert report.additional_relevant_hits == sum(
        len(q.additional_relevant_ids) for q in report.results
    )
    assert all(0 <= getattr(report.graphsage, field) <= 1 for field in type(report.graphsage).model_fields)
    assert "no es una prueba inductiva" in " ".join(report.limitations)


def test_untrained_and_missing_cohort_refuse_to_invent_results() -> None:
    with pytest.raises(BenchmarkNotReadyError, match="Faltan embeddings"):
        BenchmarkService(FakeDriver(has_vectors=False), "neo4j").run()
    with pytest.raises(BenchmarkNotReadyError, match="Cohorte BENCH-"):
        BenchmarkService(FakeDriver(incomplete=True), "neo4j").run()


def test_fixture_ids_match_scenario_and_labels_not_sent_to_neo4j() -> None:
    truth = json.loads((Path(__file__).resolve().parents[1] / "datasets/data/benchmark_truth.json").read_text())
    ids = {p["payment_id"] for p in generate_benchmark_payments()}
    assert ids == set(truth["hard_negatives"]) | {
        pid for group in truth["incident_members"].values() for pid in group
    }
    assert "incident_members" not in COHORT_CYPHER + NEURAL_CYPHER


def test_truth_is_packaged_for_installed_api_and_not_only_available_in_repo() -> None:
    # A wheel installs benchmark.py in site-packages, not under /app/src.
    # The fixture must travel inside the installed package.
    from pe.axiz.graphrag_payments.application.benchmark import TRUTH_PATH

    source_truth = Path(__file__).resolve().parents[1] / "datasets/data/benchmark_truth.json"
    assert TRUTH_PATH.name == "benchmark_truth.json"
    assert TRUTH_PATH.parent.name == "data"
    assert TRUTH_PATH.parent.parent.name == "graphrag_payments"
    assert TRUTH_PATH.is_file()
    assert TRUTH_PATH.read_bytes() == source_truth.read_bytes()
    assert load_truth() == json.loads(source_truth.read_text(encoding="utf-8"))


def test_missing_or_broken_truth_is_actionable_not_an_internal_server_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(BenchmarkNotReadyError, match="Referencia de evaluación no disponible"):
        load_truth(missing)
    invalid = tmp_path / "broken.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(BenchmarkNotReadyError, match="Referencia de evaluación no disponible"):
        load_truth(invalid)
