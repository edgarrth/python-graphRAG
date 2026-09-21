"""Conservative, explainable payment-reference routing for conversational GraphRAG.

Do not ask an LLM to invent an identifier. Similarity is only requested for an
explicit similarity intent and exactly one unambiguous reference, or for a
single reference explicitly provided by the current conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PAYMENT_PATTERN = re.compile(r"(?<![\w-])PAY-[0-9]{3,}(?![\w-])", re.IGNORECASE)
SIMILARITY_PATTERN = re.compile(
    r"\b(?:similar(?:es)?|similitud|parecid[oa]s?|semejant(?:es)?|"
    r"vecin[oa]s?|"
    r"mismo\s+patr[oó]n|patrones\s+(?:similares|parecidos)|"
    r"similarity|similar\s+payments|like\s+this\s+payment)\b",
    re.IGNORECASE,
)


def payment_ids(text: str) -> list[str]:
    """Extract de-duplicated, bounded identifiers from the user's text only."""
    return list(dict.fromkeys(match.group(0).upper() for match in PAYMENT_PATTERN.finditer(text)))


@dataclass(frozen=True)
class NeuralRoute:
    action: Literal["traditional", "neural", "needs_reference", "multiple_references"]
    payment_id: str | None = None
    note: str = ""
    from_conversation: bool = False


def resolve_neural_route(
    question: str,
    *,
    retrieval_mode: str = "traditional",
    neural_payment_id: str | None = None,
    conversation_payment_id: str | None = None,
) -> NeuralRoute:
    """The legacy explicit neural_payment_id takes priority for API compatibility."""
    if neural_payment_id:
        return NeuralRoute("neural", neural_payment_id.strip(), "Referencia explícita de API.")
    if retrieval_mode != "auto" or not SIMILARITY_PATTERN.search(question):
        return NeuralRoute("traditional", note="Recuperación GraphRAG sin búsqueda neuronal.")

    ids = payment_ids(question)
    if len(ids) > 1:
        return NeuralRoute(
            "multiple_references",
            note="Indica un único pago de referencia para buscar sus vecinos GraphSAGE.",
        )
    if ids:
        return NeuralRoute("neural", ids[0], "Pago detectado en esta pregunta.")

    # A previous, explicitly mentioned reference is only considered in the
    # same conversation; no inference based on model output or database ranking.
    previous = (conversation_payment_id or "").strip().upper()
    if previous and PAYMENT_PATTERN.fullmatch(previous):
        return NeuralRoute("neural", previous, "Pago retomado de esta conversación.", True)
    return NeuralRoute(
        "needs_reference",
        note="¿De qué pago quieres encontrar similares? Indica un ID, por ejemplo PAY-1008.",
    )
