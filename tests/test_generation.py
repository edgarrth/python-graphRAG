from pe.axiz.graphrag_payments.application.generation import DeterministicGroundedGenerator
from pe.axiz.graphrag_payments.domain.models import (
    ContextItem,
    PaymentEvidence,
    ReasonCodeEvidence,
)


def test_deterministic_generator_is_grounded() -> None:
    generator = DeterministicGroundedGenerator()
    context = ContextItem(
        chunk_id="KB-DECLINE-05",
        title="Rechazo 05 - Do not honor",
        text="El código 05 es un rechazo genérico emitido por el emisor.",
        score=0.91,
        reason_codes=[ReasonCodeEvidence(code="05", description="Do not honor")],
        related_payments=[
            PaymentEvidence(
                payment_id="PAY-1001",
                status="DECLINED",
                amount=249.9,
                currency="PEN",
                merchant="TechStore Perú",
                acquirer="Andes Acquiring",
            )
        ],
    )

    answer = generator.generate("¿Qué pasó?", [context])

    assert "Rechazo 05" in answer
    assert "PAY-1001" in answer
    assert "05" in answer
