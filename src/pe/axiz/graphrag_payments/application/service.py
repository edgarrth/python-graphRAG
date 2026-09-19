from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver
    from pe.axiz.graphrag_payments.application.retrieval import GraphAwareRetriever

from pe.axiz.graphrag_payments.application.generation import AnswerGenerator
from pe.axiz.graphrag_payments.domain.models import (
    GraphEdge,
    GraphNode,
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

    def query(self, request: GraphRagQueryRequest) -> GraphRagQueryResponse:
        top_k = request.top_k or self._settings.default_top_k
        top_k = min(top_k, self._settings.max_top_k)
        retriever = self._retriever
        if retriever is None:
            from pe.axiz.graphrag_payments.application.retrieval import GraphAwareRetriever

            retriever = GraphAwareRetriever(self._driver, self._settings)
            self._retriever = retriever
        retrieval = retriever.search(request.question, top_k)
        answer = self._generator.generate(request.question, retrieval.contexts)
        contexts = retrieval.contexts if request.include_context else []
        return GraphRagQueryResponse(
            question=request.question,
            answer=answer,
            generation_provider=self._settings.generation_provider,
            contexts=contexts,
            trace=RetrievalTrace(
                retriever="HybridCypherRetriever",
                vector_index=self._settings.vector_index_name,
                fulltext_index=self._settings.fulltext_index_name,
                graph_expansion=(
                    "KnowledgeChunk -> ReasonCode <- Payment -> Merchant / Acquirer"
                ),
                top_k=top_k,
                returned_contexts=len(retrieval.contexts),
            ),
        )

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
