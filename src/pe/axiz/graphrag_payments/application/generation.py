from __future__ import annotations

import json
from abc import ABC, abstractmethod

from pe.axiz.graphrag_payments.domain.models import ContextItem
from pe.axiz.graphrag_payments.settings import Settings


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[ContextItem]) -> str:
        raise NotImplementedError


class DeterministicGroundedGenerator(AnswerGenerator):
    """Credential-free synthesis used as the default runnable PoC mode."""

    def generate(self, question: str, contexts: list[ContextItem]) -> str:
        if not contexts:
            return "No se encontró contexto suficiente en el grafo para responder la consulta."

        best = contexts[0]
        reason_codes = sorted({reason.code for item in contexts for reason in item.reason_codes})
        payments = []
        seen: set[str] = set()
        for item in contexts:
            for payment in item.related_payments:
                if payment.payment_id not in seen:
                    seen.add(payment.payment_id)
                    payments.append(payment)

        parts = [
            f"La evidencia más relevante es '{best.title}'. {best.text}",
        ]
        if reason_codes:
            parts.append(f"Códigos relacionados en el grafo: {', '.join(reason_codes)}.")
        if payments:
            examples = ", ".join(
                f"{p.payment_id} ({p.status}, {p.amount:.2f} {p.currency}, "
                f"{p.merchant or 'sin comercio'}, {p.acquirer or 'sin adquirente'})"
                for p in payments[:3]
            )
            parts.append(f"Pagos conectados que sirven como evidencia: {examples}.")
        parts.append(
            "La respuesta se limita al contexto recuperado por búsqueda híbrida y expansión del grafo; "
            "no se agregaron hechos externos."
        )
        return " ".join(parts)


class OpenAIGroundedGenerator(AnswerGenerator):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY es obligatorio cuando GENERATION_PROVIDER=openai")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, question: str, contexts: list[ContextItem]) -> str:
        serialized_context = json.dumps(
            [context.model_dump() for context in contexts],
            ensure_ascii=False,
            indent=2,
        )
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de operaciones de pagos. Responde únicamente con la "
                        "evidencia GraphRAG entregada. Si la evidencia no alcanza, dilo explícitamente. "
                        "No inventes causas, métricas ni acciones."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Pregunta:\n{question}\n\nContexto GraphRAG:\n{serialized_context}",
                },
            ],
        )
        return response.output_text.strip()


def build_generator(settings: Settings) -> AnswerGenerator:
    if settings.generation_provider == "openai":
        return OpenAIGroundedGenerator(settings)
    return DeterministicGroundedGenerator()
