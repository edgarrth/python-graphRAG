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


def test_openai_grounding_prompt_preserves_explicit_reason_codes() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/pe/axiz/graphrag_payments/application/generation.py"
    ).read_text(encoding="utf-8")

    assert "coincida exactamente con ese código" in source
    assert "no lo sustituyas por otro código relacionado" in source


def test_deterministic_generator_respects_anchored_context_order() -> None:
    generator = DeterministicGroundedGenerator()
    code_91 = ContextItem(
        chunk_id="KB-ISSUER-91",
        title="Código 91 - Emisor o switch no disponible",
        text="El código 91 indica indisponibilidad temporal del emisor o switch.",
        score=1.0,
        reason_codes=[ReasonCodeEvidence(code="91", description="Issuer unavailable")],
    )
    code_51 = ContextItem(
        chunk_id="KB-DECLINE-51",
        title="Rechazo 51 - Fondos insuficientes",
        text="El código 51 indica fondos insuficientes.",
        score=0.8,
        reason_codes=[ReasonCodeEvidence(code="51", description="Insufficient funds")],
    )

    answer = generator.generate("¿Para qué es el código 91?", [code_91, code_51])

    assert "Código 91" in answer
    assert "indisponibilidad temporal" in answer
