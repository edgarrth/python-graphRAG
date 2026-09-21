"""Pure chat-turn planning, shared by both retrieval modalities."""
from __future__ import annotations

from typing import Any

from chat_routing import payment_ids


def plan_turn(conversation: dict[str, Any], question: str, selected_mode: str) -> dict[str, Any]:
    """Use only explicit user IDs. The previous ID applies to follow-up turns."""
    question = question.strip()
    previous_reference = conversation.get("last_payment_id")
    ids = payment_ids(question)
    if len(ids) == 1:
        conversation["last_payment_id"] = ids[0]
    elif len(ids) > 1:
        conversation["last_payment_id"] = None
    return {
        "question": question,
        "conversation_payment_id": previous_reference,
        "retrieval_mode": (
            "auto" if selected_mode == "Neural GraphRAG inteligente" else "traditional"
        ),
        "explicit_payment_count": len(ids),
    }
