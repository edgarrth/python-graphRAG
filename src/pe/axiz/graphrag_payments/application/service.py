from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver
    from pe.axiz.graphrag_payments.application.retrieval import GraphAwareRetriever

from pe.axiz.graphrag_payments.application.generation import AnswerGenerator
from pe.axiz.graphrag_payments.domain.models import (
    ContextItem,
    GraphEdge,
    GraphNode,
    GraphSageNeighbor,
    GraphRagQueryRequest,
    GraphRagQueryResponse,
    PaymentGraphResponse,
    RetrievalTrace,
    SchemaResponse,
)
from pe.axiz.graphrag_payments.settings import Settings


class GraphRagService:
    def __init__(
        self,
        driver: Driver,
        settings: Settings,
        retriever: GraphAwareRetriever | None,
        generator: AnswerGenerator,
    ) -> None:
        self._driver = driver
        self._settings = settings
        self._retriever = retriever
        self._generator = generator

    def _neural_retrieve(
        self, request: GraphRagQueryRequest, top_k: int
    ) -> tuple[Any, list[GraphSageNeighbor]]:
        """Keep exact-code and hybrid retrieval; add opt-in structural evidence."""
        retrieval = self._get_retriever().search(request.question, top_k)
        if not request.neural_payment_id:
            return retrieval, []
        from pe.axiz.graphrag_payments.application.graphsage import GraphSageService

        result = GraphSageService(self._driver, self._settings).similar(
            request.neural_payment_id, top_k=min(5, top_k)
        )
        neighbors = result.neighbors
        if neighbors:
            # A top_k=1 explicit-code query must NEVER lose its exact-code anchor.
            reserve_slot = top_k > 1 or not getattr(retrieval, "explicit_reason_codes", [])
            if not reserve_slot:
                return retrieval, neighbors
            evidence = ContextItem(
                chunk_id=f"GRAPHSAGE:{request.neural_payment_id}",
                title=f"Similitud estructural GraphSAGE de {request.neural_payment_id}",
                text=(
                    "Los vecinos enumerados fueron identificados mediante similitud coseno entre "
                    "embeddings de GraphSAGE; no prueban una causa raíz compartida, fraude ni "
                    "relación causal. Contrasta los códigos, comercios y adquirentes observados."
                ),
                score=neighbors[0].similarity,
                source="graphsage",
                neural_matches=neighbors,
            )
            # Preserve existing evidence ordering, especially exact reason-code anchors.
            retrieval = SimpleNamespace(
                contexts=[*retrieval.contexts[: max(0, top_k - 1)], evidence],
                strategy=str(getattr(retrieval, "strategy", "hybrid")) + "+graphsage",
                explicit_reason_codes=list(getattr(retrieval, "explicit_reason_codes", [])),
            )
        return retrieval, neighbors

    def _resolve_top_k(self, request: GraphRagQueryRequest) -> int:
        top_k = request.top_k or self._settings.default_top_k
        return min(top_k, self._settings.max_top_k)

    def _get_retriever(self) -> GraphAwareRetriever:
        retriever = self._retriever
        if retriever is None:
            from pe.axiz.graphrag_payments.application.retrieval import GraphAwareRetriever

            retriever = GraphAwareRetriever(self._driver, self._settings)
            self._retriever = retriever
        return retriever

    def _trace(
        self, top_k: int, retrieval: Any, payment_id: str | None = None, neural_matches: int = 0
    ) -> RetrievalTrace:
        contexts = getattr(retrieval, "contexts", [])
        return RetrievalTrace(
            retriever="HybridCypherRetriever",
            vector_index=self._settings.vector_index_name,
            fulltext_index=self._settings.fulltext_index_name,
            graph_expansion="KnowledgeChunk -> ReasonCode <- Payment -> Merchant / Acquirer",
            top_k=top_k,
            returned_contexts=len(contexts),
            ranker=self._settings.hybrid_ranker,
            vector_weight=(
                self._settings.hybrid_alpha if self._settings.hybrid_ranker == "linear" else None
            ),
            effective_search_ratio=self._settings.effective_search_ratio,
            retrieval_strategy=str(getattr(retrieval, "strategy", "hybrid")),
            explicit_reason_codes=list(getattr(retrieval, "explicit_reason_codes", [])),
            neural_payment_id=payment_id,
            neural_matches=neural_matches,
        )

    def query(self, request: GraphRagQueryRequest) -> GraphRagQueryResponse:
        top_k = self._resolve_top_k(request)
        retrieval, neighbors = self._neural_retrieve(request, top_k)
        answer = self._generator.generate(request.question, retrieval.contexts)
        contexts = retrieval.contexts if request.include_context else []
        return GraphRagQueryResponse(
            question=request.question,
            answer=answer,
            generation_provider=self._settings.generation_provider,
            generation_model=(
                self._settings.openai_model if self._settings.generation_provider == "openai" else None
            ),
            contexts=contexts,
            trace=self._trace(top_k, retrieval, request.neural_payment_id, len(neighbors)),
            neural_neighbors=neighbors,
        )

    def query_stream(self, request: GraphRagQueryRequest) -> Iterator[dict[str, Any]]:
        """Yield typed events for the SSE transport."""
        started = perf_counter()
        top_k = self._resolve_top_k(request)
        yield {
            "event": "stage",
            "data": {"stage": "retrieval", "message": "Recuperando contexto GraphRAG…"},
        }

        retrieval_started = perf_counter()
        retrieval, neighbors = self._neural_retrieve(request, top_k)
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000, 1)
        trace = self._trace(top_k, retrieval, request.neural_payment_id, len(neighbors))
        yield {
            "event": "retrieval",
            "data": {
                "message": "Contexto híbrido recuperado y grafo expandido.",
                "retrieval_ms": retrieval_ms,
                "trace": trace.model_dump(mode="json"),
                "neural_neighbors": [n.model_dump(mode="json") for n in neighbors],
                "contexts": (
                    [item.model_dump(mode="json") for item in retrieval.contexts]
                    if request.include_context
                    else []
                ),
            },
        }

        yield {
            "event": "stage",
            "data": {
                "stage": "generation",
                "message": (
                    f"Generando respuesta con {self._settings.openai_model}…"
                    if self._settings.generation_provider == "openai"
                    else "Generando respuesta determinística…"
                ),
            },
        }

        answer_parts: list[str] = []
        generation_started = perf_counter()
        for delta in self._generator.stream(request.question, retrieval.contexts):
            answer_parts.append(delta)
            yield {"event": "delta", "data": {"delta": delta}}

        answer = "".join(answer_parts).strip()
        generation_ms = round((perf_counter() - generation_started) * 1000, 1)
        total_ms = round((perf_counter() - started) * 1000, 1)
        response = GraphRagQueryResponse(
            question=request.question,
            answer=answer,
            generation_provider=self._settings.generation_provider,
            generation_model=(
                self._settings.openai_model if self._settings.generation_provider == "openai" else None
            ),
            contexts=retrieval.contexts if request.include_context else [],
            trace=trace,
            neural_neighbors=neighbors,
        )
        yield {
            "event": "complete",
            "data": {
                **response.model_dump(mode="json"),
                "timings": {
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "total_ms": total_ms,
                },
            },
        }

    def schema(self) -> SchemaResponse:
        labels, _, _ = self._driver.execute_query(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels",
            database_=self._settings.neo4j_database,
        )
        rels, _, _ = self._driver.execute_query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types",
            database_=self._settings.neo4j_database,
        )
        indexes, _, _ = self._driver.execute_query(
            "SHOW INDEXES YIELD name RETURN collect(name) AS names",
            database_=self._settings.neo4j_database,
        )
        return SchemaResponse(
            node_labels=sorted(labels[0]["labels"] if labels else []),
            relationship_types=sorted(rels[0]["types"] if rels else []),
            indexes=sorted(indexes[0]["names"] if indexes else []),
        )

    def payment_graph(self, payment_id: str) -> PaymentGraphResponse:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (payment:Payment {payment_id: $payment_id})
            OPTIONAL MATCH path=(payment)-[r]-(neighbor)
            RETURN payment,
                   collect(DISTINCT neighbor) AS neighbors,
                   collect(DISTINCT {
                     source: elementId(startNode(r)),
                     target: elementId(endNode(r)),
                     type: type(r)
                   }) AS relationships
            """,
            payment_id=payment_id,
            database_=self._settings.neo4j_database,
        )
        if not records:
            return PaymentGraphResponse(payment_id=payment_id, nodes=[], edges=[])

        record = records[0]
        payment = record["payment"]
        neighbors = [node for node in record["neighbors"] if node is not None]
        raw_nodes = [payment, *neighbors]
        nodes = [
            GraphNode(
                id=node.element_id,
                labels=list(node.labels),
                properties={key: _json_safe(value) for key, value in dict(node).items()},
            )
            for node in raw_nodes
        ]
        edges = [GraphEdge(**edge) for edge in record["relationships"] if edge.get("type")]
        return PaymentGraphResponse(payment_id=payment_id, nodes=nodes, edges=edges)


def _json_safe(value: object) -> object:
    if hasattr(value, "iso_format"):
        return value.iso_format()  # type: ignore[no-any-return, union-attr]
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return, union-attr]
    return value
