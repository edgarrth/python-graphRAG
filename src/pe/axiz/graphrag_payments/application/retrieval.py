from __future__ import annotations

import json
from dataclasses import dataclass, field

from neo4j import Driver, Record
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from pe.axiz.graphrag_payments.application.query_routing import extract_explicit_reason_codes
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

EXACT_REASON_CODE_QUERY = """
UNWIND $codes AS explicit_code
MATCH (node:KnowledgeChunk)-[:EXPLAINS]->(target:ReasonCode)
WHERE toUpper(target.code) = toUpper(explicit_code)
WITH DISTINCT node,
     CASE
       WHEN any(code IN $codes WHERE toUpper(node.title) CONTAINS toUpper(code)) THEN 2
       ELSE 1
     END AS anchor_priority
RETURN
  node.chunk_id AS chunk_id,
  node.title AS title,
  node.text AS text,
  1.0 AS score,
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
  } AS related_payments,
  anchor_priority
ORDER BY anchor_priority DESC, chunk_id ASC
LIMIT $limit
"""


@dataclass(frozen=True)
class RetrievalResult:
    contexts: list[ContextItem]
    strategy: str = "hybrid"
    explicit_reason_codes: list[str] = field(default_factory=list)


def _payload(record: Record) -> dict[str, object]:
    return {
        "chunk_id": record.get("chunk_id"),
        "title": record.get("title"),
        "text": record.get("text"),
        "score": float(record.get("score") or 0.0),
        "reason_codes": record.get("reason_codes") or [],
        "related_payments": record.get("related_payments") or [],
    }


def _formatter(record: Record) -> RetrieverResultItem:
    payload = _payload(record)
    return RetrieverResultItem(
        content=json.dumps(payload, ensure_ascii=False),
        metadata={"score": payload["score"], "chunk_id": payload["chunk_id"]},
    )


class GraphAwareRetriever:
    """Hybrid retrieval plus exact entity anchoring and graph traversal in Neo4j."""

    def __init__(self, driver: Driver, settings: Settings) -> None:
        embedder = SentenceTransformerEmbeddings(model=settings.embedding_model)
        self._driver = driver
        self._database = settings.neo4j_database
        self._retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=settings.vector_index_name,
            fulltext_index_name=settings.fulltext_index_name,
            retrieval_query=GRAPH_EXPANSION_QUERY,
            embedder=embedder,
            result_formatter=_formatter,
            neo4j_database=settings.neo4j_database,
        )
        self._ranker = settings.hybrid_ranker
        self._alpha = settings.hybrid_alpha
        self._effective_search_ratio = settings.effective_search_ratio

    def _search_exact_reason_codes(self, codes: list[str], limit: int) -> list[ContextItem]:
        if not codes or limit <= 0:
            return []
        records, _, _ = self._driver.execute_query(
            EXACT_REASON_CODE_QUERY,
            codes=codes,
            limit=limit,
            database_=self._database,
        )
        return [ContextItem.model_validate(_payload(record)) for record in records]

    def _search_hybrid(self, question: str, top_k: int) -> list[ContextItem]:
        result = self._retriever.search(
            query_text=question,
            top_k=top_k,
            effective_search_ratio=self._effective_search_ratio,
            ranker=self._ranker,
            alpha=self._alpha if self._ranker == "linear" else None,
        )
        return [ContextItem.model_validate_json(item.content) for item in result.items]

    @staticmethod
    def _merge_contexts(
        anchored: list[ContextItem], hybrid: list[ContextItem], top_k: int
    ) -> list[ContextItem]:
        merged: list[ContextItem] = []
        seen: set[str] = set()
        for context in [*anchored, *hybrid]:
            if context.chunk_id in seen:
                continue
            seen.add(context.chunk_id)
            merged.append(context)
            if len(merged) >= top_k:
                break
        return merged

    def search(self, question: str, top_k: int) -> RetrievalResult:
        explicit_codes = extract_explicit_reason_codes(question)
        hybrid_contexts = self._search_hybrid(question, top_k)

        if not explicit_codes:
            return RetrievalResult(contexts=hybrid_contexts)

        anchored_contexts = self._search_exact_reason_codes(explicit_codes, top_k)
        if not anchored_contexts:
            # Preserve normal GraphRAG behavior if a code is syntactically explicit
            # but does not exist in the graph.
            return RetrievalResult(
                contexts=hybrid_contexts,
                strategy="hybrid_explicit_code_not_found",
                explicit_reason_codes=explicit_codes,
            )

        return RetrievalResult(
            contexts=self._merge_contexts(anchored_contexts, hybrid_contexts, top_k),
            strategy="exact_reason_code_anchor+hybrid",
            explicit_reason_codes=explicit_codes,
        )
