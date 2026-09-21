from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class GraphRagQueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_context: bool = True
    neural_payment_id: str | None = Field(default=None, min_length=1, max_length=80)
    retrieval_mode: Literal["traditional", "auto"] = "traditional"
    conversation_payment_id: str | None = Field(default=None, max_length=80)


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


class GraphSageNeighbor(BaseModel):
    payment_id: str
    similarity: float
    status: str
    amount: float
    currency: str
    merchant: str | None = None
    acquirer: str | None = None
    reason_code: str | None = None


class GraphSageSimilarResponse(BaseModel):
    payment_id: str
    model_name: str
    neighbors: list[GraphSageNeighbor]


class GraphSageTrainRequest(BaseModel):
    embedding_dimension: int = Field(default=32, ge=8, le=128)
    sample_sizes: list[int] = Field(default_factory=lambda: [10, 5], min_length=1, max_length=3)
    epochs: int = Field(default=5, ge=1, le=30)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("sample_sizes")
    @classmethod
    def validate_samples(cls, value: list[int]) -> list[int]:
        if any(sample < 1 or sample > 50 for sample in value):
            raise ValueError("Cada tamaño de muestra debe estar entre 1 y 50.")
        return value


class GraphSageTrainResponse(BaseModel):
    model_name: str
    graph_name: str
    graph_nodes: int
    graph_relationships: int
    embedded_payments: int
    embedding_dimension: int
    train_millis: int
    trained_at: str
    epoch_losses: list[float]


class GraphSageStatus(BaseModel):
    ready: bool
    payment_count: int
    embedded_payments: int
    dimension: int | None = None
    model_name: str | None = None
    trained_at: str | None = None
    train_millis: int | None = None


class ContextItem(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float
    reason_codes: list[ReasonCodeEvidence] = Field(default_factory=list)
    related_payments: list[PaymentEvidence] = Field(default_factory=list)
    neural_matches: list[GraphSageNeighbor] = Field(default_factory=list)
    source: Literal["hybrid", "graphsage"] = "hybrid"


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
    neural_payment_id: str | None = None
    neural_matches: int = 0
    neural_route: str = "traditional"
    neural_note: str | None = None


class GraphRagQueryResponse(BaseModel):
    question: str
    answer: str
    generation_provider: Literal["deterministic", "openai"]
    generation_model: str | None = None
    contexts: list[ContextItem] = Field(default_factory=list)
    trace: RetrievalTrace
    neural_neighbors: list[GraphSageNeighbor] = Field(default_factory=list)


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
