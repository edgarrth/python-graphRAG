from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver
    from pe.axiz.graphrag_payments.application.retrieval import GraphAwareRetriever

from pe.axiz.graphrag_payments.application.generation import AnswerGenerator
from pe.axiz.graphrag_payments.application.neural_routing import NeuralRoute, payment_ids, resolve_neural_route
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

    @staticmethod
    def _route(request: GraphRagQueryRequest) -> NeuralRoute:
        return resolve_neural_route(
            request.question,
            retrieval_mode=request.retrieval_mode,
            neural_payment_id=request.neural_payment_id,
            conversation_payment_id=request.conversation_payment_id,
        )

    @staticmethod
    def _effective_question(request: GraphRagQueryRequest, route: NeuralRoute) -> str:
        # A follow-up like "¿y otros similares?" needs the prior explicit payment
        # reference in the retrieval and generation prompts, not only GraphSAGE.
        if route.from_conversation and route.payment_id and not payment_ids(request.question):
            return f"{request.question} (pago de referencia: {route.payment_id})"
        return request.question

    def _neural_retrieve(
        self, request: GraphRagQueryRequest, top_k: int, route: NeuralRoute
    ) -> tuple[Any, list[GraphSageNeighbor], str | None]:
        """Keep old hybrid results, add neural evidence only for resolved references."""
        retrieval = self._get_retriever().search(self._effective_question(request, route), top_k)
        if route.action != "neural" or not route.payment_id:
            return retrieval, [], None
        from pe.axiz.graphrag_payments.application.graphsage import (
            GraphSageNotReadyError,
            GraphSagePaymentNotFoundError,
            GraphSageService,
        )

        try:
            result = GraphSageService(self._driver, self._settings).similar(
                route.payment_id, top_k=min(5, top_k)
            )
        except (GraphSageNotReadyError, GraphSagePaymentNotFoundError) as exc:
            if request.retrieval_mode != "auto" or request.neural_payment_id:
                raise  # Original explicit API contract is unchanged.
            return retrieval, [], f"GraphSAGE no ejecutado para {route.payment_id}: {exc}"
        neighbors = result.neighbors
        if neighbors:
            # Keep exact-code anchor even at top_k=1.
            reserve_slot = top_k > 1 or not getattr(retrieval, "explicit_reason_codes", [])
            if not reserve_slot:
                return retrieval, neighbors, None
            evidence = ContextItem(
                chunk_id=f"GRAPHSAGE:{route.payment_id}",
                title=f"Similitud estructural GraphSAGE de {route.payment_id}",
                text=(
                    "Los vecinos enumerados fueron identificados mediante similitud coseno entre "
                    "embeddings de GraphSAGE; no prueban una causa raíz compartida, fraude ni "
                    "relación causal. Contrasta los códigos, comercios y adquirentes observados."
                ),
                score=neighbors[0].similarity,
                source="graphsage",
                neural_matches=neighbors,
            )
            retrieval = SimpleNamespace(
                contexts=[*retrieval.contexts[: max(0, top_k - 1)], evidence],
                strategy=str(getattr(retrieval, "strategy", "hybrid")) + "+graphsage",
                explicit_reason_codes=list(getattr(retrieval, "explicit_reason_codes", [])),
            )
        return retrieval, neighbors, None

    def _clarification(self, request: GraphRagQueryRequest, route: NeuralRoute) -> GraphRagQueryResponse:
        """Never invent a reference or generate pseudo-neural results for ambiguous requests."""
        retrieval = SimpleNamespace(contexts=[], strategy=route.action, explicit_reason_codes=[])
        return GraphRagQueryResponse(
            question=request.question,
            answer=route.note,
            generation_provider=self._settings.generation_provider,
            generation_model=(
                self._settings.openai_model if self._settings.generation_provider == "openai" else None
            ),
            contexts=[],
            trace=self._trace(self._resolve_top_k(request), retrieval, route=route),
            neural_neighbors=[],
        )

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
        self,
        top_k: int,
        retrieval: Any,
        payment_id: str | None = None,
        neural_matches: int = 0,
        *,
        route: NeuralRoute | None = None,
        neural_warning: str | None = None,
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
            neural_route=("unavailable" if neural_warning else route.action if route else "traditional"),
            neural_note=neural_warning or (route.note if route else None),
        )

    def query(self, request: GraphRagQueryRequest) -> GraphRagQueryResponse:
        route = self._route(request)
        if route.action in ("needs_reference", "multiple_references"):
            return self._clarification(request, route)
        top_k = self._resolve_top_k(request)
        retrieval, neighbors, warning = self._neural_retrieve(request, top_k, route)
        answer = self._generator.generate(self._effective_question(request, route), retrieval.contexts)
        contexts = retrieval.contexts if request.include_context else []
        return GraphRagQueryResponse(
            question=request.question,
            answer=answer,
            generation_provider=self._settings.generation_provider,
            generation_model=(
                self._settings.openai_model if self._settings.generation_provider == "openai" else None
            ),
            contexts=contexts,
            trace=self._trace(
                top_k, retrieval, route.payment_id, len(neighbors), route=route,
                neural_warning=warning,
            ),
            neural_neighbors=neighbors,
        )

    def query_stream(self, request: GraphRagQueryRequest) -> Iterator[dict[str, Any]]:
        """REST and SSE share the same deterministic routing decisions."""
        started = perf_counter()
        route = self._route(request)
        if route.action in ("needs_reference", "multiple_references"):
            response = self._clarification(request, route)
            yield {"event": "stage", "data": {"stage": "routing", "message": "Aclarando pago de referencia…"}}
            yield {"event": "delta", "data": {"delta": response.answer}}
            yield {"event": "complete", "data": {
                **response.model_dump(mode="json"),
                "timings": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": 0},
            }}
            return
        top_k = self._resolve_top_k(request)
        yield {
            "event": "stage",
            "data": {"stage": "retrieval", "message": "Recuperando contexto GraphRAG…"},
        }

        retrieval_started = perf_counter()
        retrieval, neighbors, warning = self._neural_retrieve(request, top_k, route)
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000, 1)
        trace = self._trace(
            top_k, retrieval, route.payment_id, len(neighbors), route=route,
            neural_warning=warning,
        )
        yield {
            "event": "retrieval",
            "data": {
                "message": "Contexto híbrido recuperado y grafo expandido.",
                "retrieval_ms": retrieval_ms,
                "trace": trace.model_dump(mode="json"),
                "neural_neighbors": [n.model_dump(mode="json") for n in neighbors],
                "contexts": (
                    [item.model_dump(mode="json") for item in retrieval.contexts]
                    if request.include_context else []
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
        for delta in self._generator.stream(self._effective_question(request, route), retrieval.contexts):
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
        yield {"event": "complete", "data": {
            **response.model_dump(mode="json"),
            "timings": {
                "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        }}

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
