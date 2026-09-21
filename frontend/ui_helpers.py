from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


EXAMPLE_QUESTIONS = [
    "¿Por qué se rechazan pagos con código 05 y qué debería revisar operaciones?",
    "¿Qué evidencia hay sobre fondos insuficientes y qué pagos están relacionados?",
    "¿Qué puede causar timeouts del emisor o adquirente en el flujo de pagos?",
    "¿Cómo ayuda la idempotencia a evitar cobros duplicados?",
    "Busca pagos similares a PAY-1008 y explica sus relaciones.",
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
        ("Estrategia", str(trace.get("retrieval_strategy", "hybrid"))),
        ("Enrutamiento neuronal", str(trace.get("neural_route", "traditional"))),
        ("Pago neuronal", str(trace.get("neural_payment_id") or "—")),
        ("Detalle neuronal", str(trace.get("neural_note") or "—")),
        ("Códigos anclados", ", ".join(trace.get("explicit_reason_codes") or []) or "—"),
        ("Contextos retornados", str(trace.get("returned_contexts", 0))),
    ]


def graphsage_usage(trace: dict[str, Any] | None) -> str:
    """Always-visible, truthful summary of actual neural execution."""
    trace = trace or {}
    route = trace.get("neural_route", "traditional")
    note = str(trace.get("neural_note") or "").strip()
    if route == "neural":
        payment_id = str(trace.get("neural_payment_id") or "—")
        matches = int(trace.get("neural_matches") or 0)
        return (
            f"GraphSAGE: Sí · Referencia {payment_id} · {matches} pagos similares. "
            "La similitud estructural no demuestra causalidad."
        )
    if route == "unavailable":
        return f"GraphSAGE: No · {note or 'Modelo no disponible; se utilizó GraphRAG tradicional.'}"
    if route in ("needs_reference", "multiple_references"):
        return f"GraphSAGE: No · {note or 'Se requiere aclarar el pago de referencia.'}"
    return f"GraphSAGE: No · {note or 'Se utilizó únicamente GraphRAG tradicional.'}"
