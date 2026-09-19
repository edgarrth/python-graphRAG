from __future__ import annotations

import json
from dataclasses import dataclass

from neo4j import Driver, Record
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from pe.axiz.graphrag_payments.domain.models import ContextItem
from pe.axiz.graphrag_payments.settings import Settings


GRAPH_EXPANSION_QUERY = """
RETURN
  node.chunk_id AS chunk_id,
  node.title AS title,
  node.text AS text,
  score AS score,
  collect {
    MATCH (node)-[:EXPLAINS]->(reason:ReasonCode)
    RETURN {code: reason.code, description: reason.description}
  } AS reason_codes,
  collect {
    MATCH (node)-[:EXPLAINS]->(reason:ReasonCode)<-[:FAILED_WITH]-(payment:Payment)
    OPTIONAL MATCH (payment)-[:AT_MERCHANT]->(merchant:Merchant)
    OPTIONAL MATCH (payment)-[:ROUTED_TO]->(acquirer:Acquirer)
    RETURN {
      payment_id: payment.payment_id,
      status: payment.status,
      amount: payment.amount,
      currency: payment.currency,
      merchant: merchant.name,
      acquirer: acquirer.name,
      created_at: toString(payment.created_at)
    }
    ORDER BY payment.created_at DESC
    LIMIT 5
  } AS related_payments
"""


@dataclass(frozen=True)
class RetrievalResult:
    contexts: list[ContextItem]


def _formatter(record: Record) -> RetrieverResultItem:
    payload = {
        "chunk_id": record.get("chunk_id"),
        "title": record.get("title"),
        "text": record.get("text"),
        "score": float(record.get("score") or 0.0),
        "reason_codes": record.get("reason_codes") or [],
        "related_payments": record.get("related_payments") or [],
    }
    return RetrieverResultItem(
        content=json.dumps(payload, ensure_ascii=False),
        metadata={"score": payload["score"], "chunk_id": payload["chunk_id"]},
    )


class GraphAwareRetriever:
    """Hybrid semantic/lexical retrieval followed by graph traversal in Neo4j."""

    def __init__(self, driver: Driver, settings: Settings) -> None:
        embedder = SentenceTransformerEmbeddings(model=settings.embedding_model)
        self._retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=settings.vector_index_name,
            fulltext_index_name=settings.fulltext_index_name,
            retrieval_query=GRAPH_EXPANSION_QUERY,
            embedder=embedder,
            result_formatter=_formatter,
            neo4j_database=settings.neo4j_database,
        )

    def search(self, question: str, top_k: int) -> RetrievalResult:
        result = self._retriever.search(query_text=question, top_k=top_k)
        contexts: list[ContextItem] = []
        for item in result.items:
            contexts.append(ContextItem.model_validate_json(item.content))
        return RetrievalResult(contexts=contexts)
