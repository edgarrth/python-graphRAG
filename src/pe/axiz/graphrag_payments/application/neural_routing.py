"""Conservative, explainable payment-reference routing for conversational GraphRAG.

Do not ask an LLM to invent an identifier. Similarity is only requested for an
explicit similarity intent and exactly one unambiguous reference, or for a
single reference explicitly provided by the current conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PAYMENT_PATTERN = re.compile(r"(?<![\w-])(?:PAY|SYN)-[0-9]{3,}(?![\w-])", re.IGNORECASE)
SIMILARITY_PATTERN = re.compile(
    r"\b(?:similar(?:es)?|similitud|parecid[oa]s?|semejant(?:es)?|"
    r"vecin[oa]s?|"
    r"mismo\s+patr[oó]n|patrones\s+(?:similares|parecidos)|"
    r"similarity|similar\s+payments|like\s+this\s+payment)\b",
    re.IGNORECASE,
)
# "Pagos relacionados" is ambiguous: it can mean graph links or similar
# structure. In auto mode, return BOTH the original graph evidence and
# GraphSAGE neighbors when the query asks for related payments. Do not treat
# every occurrence of "relación" as a request for neural similarity.
# Permit intervening words ("pagos que están relacionados") and a common
# chat typo ("pags"). Resolve intent on the backend; the payment reference
# shown by Streamlit alone does not activate GraphSAGE.
_PAYMENT_NOUN = r"(?:pagos?|pags|pagoso|transacciones?|operaciones?)"
_RELATED_WORD = (
    r"(?:relacionad[oa]s?|relacionan|vinculad[oa]s?|"
    r"asociad[oa]s?|conectad[oa]s?)"
)
RELATED_PAYMENTS_PATTERN = re.compile(
    rf"\b{_PAYMENT_NOUN}\b.{{0,120}}\b{_RELATED_WORD}\b|"
    rf"\b{_RELATED_WORD}\b.{{0,120}}\b{_PAYMENT_NOUN}\b",
    re.IGNORECASE | re.DOTALL,
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
    if retrieval_mode != "auto":
        return NeuralRoute("traditional", note="Modo GraphRAG tradicional seleccionado.")
    similarity_requested = bool(SIMILARITY_PATTERN.search(question))
    related_requested = bool(RELATED_PAYMENTS_PATTERN.search(question))
    if not (similarity_requested or related_requested):
        return NeuralRoute("traditional", note="La pregunta no solicita similitud ni pagos relacionados.")

    ids = payment_ids(question)
    # "Relacionados" without a single payment (or a prior reference) may ask
    # about graph links to a code, merchant, etc. Preserve ordinary GraphRAG.
    previous = (conversation_payment_id or "").strip().upper()
    has_previous = bool(PAYMENT_PATTERN.fullmatch(previous))
    if related_requested and not similarity_requested and not ids and not has_previous:
        return NeuralRoute("traditional", note="Consulta general de relaciones: se utilizó GraphRAG.")
    if related_requested and not similarity_requested and len(ids) > 1:
        return NeuralRoute("traditional", note="Varios pagos: se consultan sus relaciones en el grafo.")
    if len(ids) > 1:
        return NeuralRoute(
            "multiple_references",
            note="Indica un único pago de referencia para buscar sus vecinos GraphSAGE.",
        )
    if ids:
        return NeuralRoute("neural", ids[0], "Pago detectado en esta pregunta.")

    # A previous, explicitly mentioned reference is only considered in the
    # same conversation; no inference based on model output or database ranking.
    if has_previous:
        return NeuralRoute("neural", previous, "Pago retomado de esta conversación.", True)
    return NeuralRoute(
        "needs_reference",
        note="¿De qué pago quieres encontrar similares? Indica un ID, por ejemplo PAY-1008 o SYN-00304.",
    )
