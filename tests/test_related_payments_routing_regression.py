"""Pure routing tests that do not require Neo4j or an LLM.

The typo below is copied from the reported chat message. Detection of an ID
for conversation state is not, on its own, a request to run GraphSAGE.
"""

import pytest

from pe.axiz.graphrag_payments.application.neural_routing import resolve_neural_route


@pytest.mark.parametrize("question", [
    "dime que pags estan relacionados a SYN-00304",
    "dime que pagos estan relacionados a SYN-00304",
    "dime qué pagos están relacionados a SYN-00304",
    "¿Qué pagos que están relacionados con SYN-00304 existen?",
    "Busca transacciones vinculadas a SYN-00304",
])
def test_related_payment_variants_activate_graphsage(question: str) -> None:
    route = resolve_neural_route(question, retrieval_mode="auto")
    assert (route.action, route.payment_id) == ("neural", "SYN-00304")


def test_related_payment_uses_current_explicit_id_not_stale_reference() -> None:
    route = resolve_neural_route(
        "dime que pags estan relacionados a SYN-00304",
        retrieval_mode="auto",
        conversation_payment_id="PAY-1008",
    )
    assert (route.action, route.payment_id) == ("neural", "SYN-00304")


def test_traditional_mode_does_not_activate_graphsage() -> None:
    route = resolve_neural_route(
        "dime que pags estan relacionados a SYN-00304", retrieval_mode="traditional"
    )
    assert route.action == "traditional"


@pytest.mark.parametrize("question", [
    "¿Qué pagos están relacionados con el código 05?",
    "¿Qué es la idempotencia?",
    "¿Qué relación tiene SYN-00304 con su comercio?",
])
def test_general_and_ordinary_graph_questions_stay_traditional(question: str) -> None:
    route = resolve_neural_route(question, retrieval_mode="auto")
    assert route.action == "traditional"
