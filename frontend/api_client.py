from __future__ import annotations

import os
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

    def payment_graph(self, payment_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/api/v1/payments/{payment_id}/graph",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
