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

    def query(self, question: str, top_k: int) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/v1/graphrag/query",
            json={"question": question, "top_k": top_k, "include_context": True},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def query_stream(self, question: str, top_k: int) -> Iterator[dict[str, Any]]:
        """Consume the API Server-Sent Events endpoint and yield typed events."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/v1/graphrag/query/stream",
            json={"question": question, "top_k": top_k, "include_context": True},
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
