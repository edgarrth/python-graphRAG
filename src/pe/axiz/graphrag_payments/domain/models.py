from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphRagQueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_context: bool = True


class PaymentEvidence(BaseModel):
    payment_id: str
    status: str
    amount: float
    currency: str
    merchant: str | None = None
    acquirer: str | None = None
    created_at: str | None = None


class ReasonCodeEvidence(BaseModel):
    code: str
    description: str


class ContextItem(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float
    reason_codes: list[ReasonCodeEvidence] = Field(default_factory=list)
    related_payments: list[PaymentEvidence] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    retriever: str
    vector_index: str
    fulltext_index: str
    graph_expansion: str
    top_k: int
    returned_contexts: int
    ranker: str = "naive"
    vector_weight: float | None = None
    effective_search_ratio: int = 1
    retrieval_strategy: str = "hybrid"
    explicit_reason_codes: list[str] = Field(default_factory=list)


class GraphRagQueryResponse(BaseModel):
    question: str
    answer: str
    generation_provider: Literal["deterministic", "openai"]
    generation_model: str | None = None
    contexts: list[ContextItem] = Field(default_factory=list)
    trace: RetrievalTrace


class GraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class PaymentGraphResponse(BaseModel):
    payment_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class SchemaResponse(BaseModel):
    node_labels: list[str]
    relationship_types: list[str]
    indexes: list[str]
