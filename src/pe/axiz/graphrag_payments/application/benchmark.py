"""Reproducible *retrieval* benchmark, not a fraud/incident classifier.

Truth stays in a local JSON fixture, never in Neo4j or GDS features. Run after
loading the benchmark cohort and training the REAL GraphSAGE GDS model.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from pe.axiz.graphrag_payments.domain.models import (
    BenchmarkCandidate,
    BenchmarkMetrics,
    BenchmarkQueryResult,
    BenchmarkResponse,
)

# The backend is installed into site-packages in Docker: parents[5] does NOT
# point to /app, so relying on the repository layout caused HTTP 500.
# Ship the reference JSON with the installed Python package instead.
TRUTH_PATH = Path(__file__).resolve().parents[1] / "data/benchmark_truth.json"

# Deliberately isolate the fixed synthetic evaluation cohort; never benchmark
# unlabeled original/SYN payments as if ground truth existed for them.
COHORT_CYPHER = """
MATCH (p:Payment)
WHERE p.payment_id STARTS WITH 'BENCH-'
OPTIONAL MATCH (p)-[:AT_MERCHANT]->(m:Merchant)
OPTIONAL MATCH (p)-[:ROUTED_TO]->(a:Acquirer)
OPTIONAL MATCH (p)-[:FAILED_WITH]->(r:ReasonCode)
OPTIONAL MATCH (c:Customer)-[:INITIATED]->(p)
OPTIONAL MATCH (p)-[:USES]->(method:PaymentMethod)
RETURN p.payment_id AS payment_id, p.amount AS amount, p.status AS status,
       toString(p.created_at) AS created_at, m.merchant_id AS merchant_id,
       a.acquirer_id AS acquirer_id, r.code AS reason_code,
       c.customer_id AS customer_id, method.method_id AS method_id,
       p.sage_embedding IS NOT NULL AS embedded
ORDER BY payment_id
"""

# Scores come from Neo4j's actual cosine function over actual GDS embeddings,
# not from a synthetic or pre-calculated imitation of GraphSAGE.
NEURAL_CYPHER = """
MATCH (source:Payment) WHERE source.payment_id IN $anchors
MATCH (candidate:Payment)
WHERE candidate.payment_id STARTS WITH 'BENCH-'
  AND candidate <> source
  AND source.sage_embedding IS NOT NULL AND candidate.sage_embedding IS NOT NULL
RETURN source.payment_id AS anchor, candidate.payment_id AS candidate,
       vector.similarity.cosine(source.sage_embedding, candidate.sage_embedding) AS score
"""


class BenchmarkNotReadyError(RuntimeError):
    """Missing cohort, labels, or complete trained GraphSAGE embeddings."""


def load_truth(path: Path | None = None) -> dict[str, Any]:
    path = path or TRUTH_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkNotReadyError(
            "Referencia de evaluación no disponible. Reconstruye la API con el ZIP actualizado."
        ) from exc
    groups = data["incident_members"]
    anchors = data["anchors"]
    members = [item for values in groups.values() for item in values]
    all_ids = set(members) | set(data["hard_negatives"])
    if (len(all_ids) != len(members) + len(data["hard_negatives"]) or
            len(anchors) != len(set(anchors)) or
            not all(anchor in members for anchor in anchors)):
        raise BenchmarkNotReadyError("La referencia de relevancia del benchmark es inconsistente.")
    return data


def business_score(source: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, list[str]]:
    """Pre-registered, non-learned relationship and operational business baseline.

    No access to hidden group identifiers, benchmark truth, or neural vectors.
    All candidates are eligible, including zero-score ones (fixed retrieval budget).
    """
    score = 0.0
    reasons: list[str] = []
    for prop, weight, name in (
        ("merchant_id", 5.0, "mismo comercio"),
        ("acquirer_id", 4.0, "mismo adquirente"),
        ("reason_code", 3.0, "mismo código de rechazo"),
        ("customer_id", 2.0, "mismo cliente"),
        ("method_id", 2.0, "mismo método de pago"),
    ):
        if source.get(prop) is not None and source[prop] == candidate.get(prop):
            score += weight
            reasons.append(name)
    t0, t1 = source.get("created_at"), candidate.get("created_at")
    if t0 and t1:
        minutes = abs((datetime.fromisoformat(t0) - datetime.fromisoformat(t1)).total_seconds()) / 60
        if minutes <= 15:
            score += 2
            reasons.append("ocurrencia dentro de 15 minutos")
    if source.get("status") and source["status"] == candidate.get("status"):
        score += 1
        reasons.append("mismo estado")
    a, b = source.get("amount"), candidate.get("amount")
    if a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0.1):
        score += 1
        reasons.append("montos dentro de ±10 %")
    return score, reasons


def ranking_metrics(retrieved: list[str], relevant: set[str], k: int) -> BenchmarkMetrics:
    """Fixed-budget precision, recall, hit-rate and binary nDCG for ONE query."""
    hits = [int(payment_id in relevant) for payment_id in retrieved[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return BenchmarkMetrics(
        precision_at_k=sum(hits) / k,
        recall_at_k=sum(hits) / len(relevant) if relevant else 0.0,
        hit_at_k=float(any(hits)),
        ndcg_at_k=dcg / ideal if ideal else 0.0,
    )


def mean_metrics(results: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    return BenchmarkMetrics(**{
        key: sum(getattr(item, key) for item in results) / len(results)
        for key in BenchmarkMetrics.model_fields
    })


def _ranked_candidates(
    ids: list[str], scores: dict[str, float], evidence: dict[str, list[str]],
    relevant: set[str], k: int,
) -> list[BenchmarkCandidate]:
    return [BenchmarkCandidate(
        payment_id=pid, score=round(scores[pid], 6),
        evidence=evidence.get(pid, []), relevant_in_synthetic_truth=pid in relevant,
    ) for pid in ids[:k]]


class BenchmarkService:
    def __init__(self, driver: Any, database: str) -> None:
        self.driver = driver
        self.database = database

    def _run(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            statement, database_=self.database, **parameters,
        )
        return [dict(record) for record in records]

    def run(self, *, top_k: int = 5) -> BenchmarkResponse:
        truth = load_truth()
        anchors: list[str] = truth["anchors"]
        members: dict[str, list[str]] = truth["incident_members"]
        relevant_by_id = {pid: group for group, ids in members.items() for pid in ids}
        expected = set(relevant_by_id) | set(truth["hard_negatives"])
        observed = self._run(COHORT_CYPHER)
        rows = {row["payment_id"]: row for row in observed}
        if set(rows) != expected:
            raise BenchmarkNotReadyError(
                "Cohorte BENCH- incompleta o distinta a la referencia: "
                f"esperados {len(expected)}, encontrados {len(rows)}. "
                "Ejecuta docker compose up -d --build para cargar el dataset del ZIP."
            )
        if not all(row["embedded"] for row in observed):
            raise BenchmarkNotReadyError(
                "Faltan embeddings GraphSAGE para pagos BENCH-. "
                "Ejecuta Entrenar GraphSAGE después de cargar el dataset."
            )
        scored = self._run(NEURAL_CYPHER, anchors=anchors)
        neural_by_anchor: dict[str, dict[str, float]] = {pid: {} for pid in anchors}
        for item in scored:
            anchor, pid, score = item["anchor"], item["candidate"], item["score"]
            if anchor not in neural_by_anchor or pid not in expected or score is None:
                continue
            neural_by_anchor[anchor][pid] = float(score)
        if any(len(scores) != len(expected) - 1 for scores in neural_by_anchor.values()):
            raise BenchmarkNotReadyError(
                "La similitud coseno neuronal no produjo un score para cada par de pagos BENCH-."
            )

        results: list[BenchmarkQueryResult] = []
        all_new: set[str] = set()
        for anchor in anchors:
            group = relevant_by_id[anchor]
            relevant = set(members[group]) - {anchor}
            baseline_scores: dict[str, float] = {}
            evidence: dict[str, list[str]] = {}
            for pid, row in rows.items():
                if pid != anchor:
                    baseline_scores[pid], evidence[pid] = business_score(rows[anchor], row)
            baseline = sorted(baseline_scores, key=lambda pid: (-baseline_scores[pid], pid))
            neural_scores = neural_by_anchor[anchor]
            neural = sorted(neural_scores, key=lambda pid: (-neural_scores[pid], pid))
            baseline_k, neural_k = baseline[:top_k], neural[:top_k]
            novel = [pid for pid in neural_k if pid in relevant and pid not in baseline_k]
            all_new.update(novel)
            # The 2K union uses a LARGER retrieval budget: report separately,
            # never compare its recall to @K as if both had equal cost.
            union_2k = list(dict.fromkeys(baseline_k + neural_k))
            results.append(BenchmarkQueryResult(
                anchor=anchor,
                incident=group,
                relevant_total=len(relevant),
                baseline=_ranked_candidates(
                    baseline, baseline_scores, evidence, relevant, top_k,
                ),
                graphsage=_ranked_candidates(
                    neural, neural_scores, {}, relevant, top_k,
                ),
                additional_relevant_ids=novel,
                baseline_metrics=ranking_metrics(baseline_k, relevant, top_k),
                graphsage_metrics=ranking_metrics(neural_k, relevant, top_k),
                union_metrics=ranking_metrics(union_2k, relevant, 2 * top_k),
            ))
        return BenchmarkResponse(
            case="Fallos operativos simulados en varios comercios/adquirentes con falsos positivos cercanos",
            synthetic=True,
            truth_policy="Los grupos de incidentes están solo en benchmark_truth.json; no se cargan como atributos/relaciones ni se usan para entrenar o recuperar.",
            cohort_payments=len(rows), queries=len(anchors), top_k=top_k,
            baseline_description="Coincidencia explícita de comercio, adquirente, código, cliente, método; ±15 min, estado y ±10 % monto. Pesos fijos; desempate por ID.",
            neural_description="GraphSAGE GDS previamente entrenado: similitud coseno de embeddings de pagos, sin usar etiquetas de incidente.",
            baseline=mean_metrics([r.baseline_metrics for r in results]),
            graphsage=mean_metrics([r.graphsage_metrics for r in results]),
            union_at_2k=mean_metrics([r.union_metrics for r in results]),
            additional_relevant_hits=sum(len(r.additional_relevant_ids) for r in results),
            distinct_additional_relevant=len(all_new),
            queries_with_additional_relevant=sum(bool(r.additional_relevant_ids) for r in results),
            results=results,
            limitations=[
                "Los 48 pagos y sus etiquetas son ficticios; las etiquetas describen intervenciones simuladas, no causas reales verificadas.",
                "Los pesos de la regla son una línea base de demostración y no se optimizan con las etiquetas.",
                "Solo se comparan candidatos BENCH-; no es una evaluación del RAG textual ni de la respuesta generada por el LLM.",
                "GDS se entrena sin etiquetas de incidente, pero ve la topología sin etiquetar de los nodos evaluados: no es una prueba inductiva en un grafo futuro.",
                "Los hallazgos adicionales @K no demuestran una mejora global: revisar también precisión, recall, nDCG y negativos difíciles.",
                "La unión usa hasta 2K candidatos y no es comparable directamente con el presupuesto @K.",
            ],
        )
