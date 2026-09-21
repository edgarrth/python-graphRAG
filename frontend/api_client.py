from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx


class ApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

    def ready(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/health/ready", timeout=10)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _query_payload(
        question: str, top_k: int, neural_payment_id: str | None = None,
        *, retrieval_mode: str = "traditional", conversation_payment_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question, "top_k": top_k, "include_context": True
        }
        if neural_payment_id:
            payload["neural_payment_id"] = neural_payment_id
        if retrieval_mode == "auto":
            payload["retrieval_mode"] = "auto"
            if conversation_payment_id:
                payload["conversation_payment_id"] = conversation_payment_id
        return payload

    def query(
        self, question: str, top_k: int, neural_payment_id: str | None = None,
        *, retrieval_mode: str = "traditional", conversation_payment_id: str | None = None,
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/v1/graphrag/query",
            json=self._query_payload(
                question, top_k, neural_payment_id,
                retrieval_mode=retrieval_mode, conversation_payment_id=conversation_payment_id,
            ),
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def query_stream(
        self, question: str, top_k: int, neural_payment_id: str | None = None,
        *, retrieval_mode: str = "traditional", conversation_payment_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Consume the original SSE endpoint, optionally enriching with GraphSAGE."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/v1/graphrag/query/stream",
            json=self._query_payload(
                question, top_k, neural_payment_id,
                retrieval_mode=retrieval_mode, conversation_payment_id=conversation_payment_id,
            ),
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            timeout=httpx.Timeout(180.0, connect=10.0),
        ) as response:
            response.raise_for_status()
            event_name = "message"
            data_lines: list[str] = []

            for line in response.iter_lines():
                if line == "":
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        yield {
                            "event": event_name,
                            "data": json.loads(raw_data) if raw_data else {},
                        }
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                raw_data = "\n".join(data_lines)
                yield {
                    "event": event_name,
                    "data": json.loads(raw_data) if raw_data else {},
                }

    def payment_graph(self, payment_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/api/v1/payments/{payment_id}/graph",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def graphsage_status(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/api/v1/graphsage/status", timeout=15)
        response.raise_for_status()
        return response.json()

    def graphsage_train(self, epochs: int, embedding_dimension: int) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/v1/graphsage/train",
            json={"epochs": epochs, "embedding_dimension": embedding_dimension},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def graphsage_similar(self, payment_id: str, top_k: int = 5) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/api/v1/graphsage/payments/{payment_id}/similar",
            params={"top_k": top_k},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
