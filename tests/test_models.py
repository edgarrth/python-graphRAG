import pytest
from pydantic import ValidationError

from pe.axiz.graphrag_payments.domain.models import GraphRagQueryRequest


def test_query_request_validates_question_length() -> None:
    with pytest.raises(ValidationError):
        GraphRagQueryRequest(question="x")


def test_query_request_accepts_valid_payload() -> None:
    request = GraphRagQueryRequest(question="¿Por qué falla PAY-1003?", top_k=3)
    assert request.top_k == 3
