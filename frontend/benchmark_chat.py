"""Chat command and compact presentation helpers for the synthetic benchmark.

Keep the experiment a *deliberate* action; a regular question mentioning
GraphSAGE must keep the usual GraphRAG conversational routing.
"""
from __future__ import annotations

import re
from typing import Any

BENCHMARK_EXAMPLE_QUESTION = "¿GraphSAGE aporta pagos relevantes? (K=5)"


def benchmark_top_k(question: str) -> int | None:
    """Return the requested K only for the dedicated example or slash command."""
    normalized = " ".join(question.strip().split())
    if normalized == BENCHMARK_EXAMPLE_QUESTION:
        return 5
    match = re.fullmatch(r"/evaluar\s+graphsage(?:\s+k\s*=\s*(\d{1,2}))?", normalized, re.IGNORECASE)
    if match:
        top_k = int(match.group(1) or 5)
        return top_k if 1 <= top_k <= 20 else None
    return None


def strategy_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    k = report["top_k"]
    return [
        {
            "Estrategia": label,
            "Precisión": f"{metrics['precision_at_k']:.1%}",
            "Recall": f"{metrics['recall_at_k']:.1%}",
            "nDCG": f"{metrics['ndcg_at_k']:.3f}",
        }
        for label, metrics in (
            (f"Reglas @{k}", report["baseline"]),
            (f"GraphSAGE @{k}", report["graphsage"]),
            (f"Unión @≤{2 * k}*", report["union_at_2k"]),
        )
    ]


def query_rows(query: dict[str, Any]) -> list[dict[str, str]]:
    """One full-width result table, not two squashed side-by-side tables."""
    return [
        {
            "Método": label,
            "Pago": str(candidate["payment_id"]),
            "Puntaje": f"{candidate['score']:.3f}",
            "Relevante": "Sí" if candidate["relevant_in_synthetic_truth"] else "No",
        }
        for label, candidates in (("Reglas", query["baseline"]), ("GraphSAGE", query["graphsage"]))
        for candidate in candidates
    ]


def additional_payment_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    """Exact relevant candidate IDs that only GraphSAGE retrieved within K."""
    return [
        {"Consulta": str(item["anchor"]), "Pago adicional": str(payment_id)}
        for item in report["results"]
        for payment_id in item["additional_relevant_ids"]
    ]
