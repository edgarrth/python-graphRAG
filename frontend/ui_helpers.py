from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


EXAMPLE_QUESTIONS = [
    "¿Por qué se rechazan pagos con código 05 y qué debería revisar operaciones?",
    "¿Qué evidencia hay sobre fondos insuficientes y qué pagos están relacionados?",
    "¿Qué puede causar timeouts del emisor o adquirente en el flujo de pagos?",
    "¿Cómo ayuda la idempotencia a evitar cobros duplicados?",
]


def conversation_title(question: str, max_length: int = 42) -> str:
    """Create a compact sidebar title from the first user question."""
    normalized = " ".join(question.strip().split())
    if not normalized:
        return "Nueva conversación"
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def conversation_group(updated_at: datetime, now: datetime | None = None) -> str:
    """Group conversations using the same mental model as modern chat UIs."""
    reference = now or datetime.now(tz=updated_at.tzinfo)
    updated_date = updated_at.date()
    reference_date = reference.date()

    if updated_date == reference_date:
        return "Hoy"
    if updated_date == reference_date - timedelta(days=1):
        return "Ayer"
    if updated_at >= reference - timedelta(days=7):
        return "Últimos 7 días"
    if updated_at >= reference - timedelta(days=30):
        return "Últimos 30 días"
    return "Anteriores"


def trace_rows(trace: dict[str, Any] | None) -> list[tuple[str, str]]:
    trace = trace or {}
    return [
        ("Retriever", str(trace.get("retriever", "—"))),
        ("Índice vectorial", str(trace.get("vector_index", "—"))),
        ("Índice full-text", str(trace.get("fulltext_index", "—"))),
        ("Expansión del grafo", str(trace.get("graph_expansion", "—"))),
        ("Ranker híbrido", str(trace.get("ranker", "—"))),
        ("Peso vectorial", str(trace.get("vector_weight", "—"))),
        ("Search ratio", str(trace.get("effective_search_ratio", "—"))),
        ("Contextos retornados", str(trace.get("returned_contexts", 0))),
    ]
