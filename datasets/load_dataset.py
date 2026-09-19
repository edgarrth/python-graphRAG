from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.sentence_transformers import SentenceTransformerEmbeddings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "data"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "axiz-graphrag-poc")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "knowledge_embedding")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME", "knowledge_fulltext")

REASON_DESCRIPTIONS = {
    "05": "Do not honor / rechazo genérico del emisor",
    "51": "Fondos insuficientes",
    "91": "Emisor o switch no disponible",
    "3DS_TIMEOUT": "Timeout durante autenticación 3DS",
    "DUPLICATE_RISK": "Riesgo de duplicidad por retry sin idempotencia",
}


def load_json(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def wait_for_neo4j(driver: Any, retries: int = 60) -> None:
    for attempt in range(1, retries + 1):
        try:
            driver.verify_connectivity()
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2)


def create_schema(driver: Any, embedding_dimension: int) -> None:
    statements = [
        "CREATE CONSTRAINT payment_id IF NOT EXISTS FOR (n:Payment) REQUIRE n.payment_id IS UNIQUE",
        "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE",
        "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (n:Merchant) REQUIRE n.merchant_id IS UNIQUE",
        "CREATE CONSTRAINT acquirer_id IF NOT EXISTS FOR (n:Acquirer) REQUIRE n.acquirer_id IS UNIQUE",
        "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (n:PaymentMethod) REQUIRE n.method_id IS UNIQUE",
        "CREATE CONSTRAINT reason_code IF NOT EXISTS FOR (n:ReasonCode) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:KnowledgeChunk) REQUIRE n.chunk_id IS UNIQUE",
        f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS FOR (n:KnowledgeChunk) ON (n.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimension}, "
        "`vector.similarity_function`: 'cosine'}}}",
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
        "FOR (n:KnowledgeChunk) ON EACH [n.search_text]",
    ]
    for statement in statements:
        driver.execute_query(statement, database_=NEO4J_DATABASE)


def wait_for_indexes(driver: Any, index_names: list[str], retries: int = 60) -> None:
    """Wait until the GraphRAG indexes are ONLINE before the API can consume them."""
    for attempt in range(1, retries + 1):
        records, _, _ = driver.execute_query(
            """
            SHOW INDEXES YIELD name, state
            WHERE name IN $index_names
            RETURN name, state
            """,
            index_names=index_names,
            database_=NEO4J_DATABASE,
        )
        states = {record["name"]: record["state"] for record in records}
        if all(states.get(name) == "ONLINE" for name in index_names):
            return
        if attempt == retries:
            raise RuntimeError(f"Indexes did not become ONLINE: {states}")
        time.sleep(1)


def load_knowledge(driver: Any, embedder: SentenceTransformerEmbeddings) -> None:
    for item in load_json("knowledge.json"):
        search_text = f"{item['title']}. {item['text']}"
        embedding = embedder.embed_query(search_text)
        driver.execute_query(
            """
            MERGE (chunk:KnowledgeChunk {chunk_id: $chunk_id})
            SET chunk.title = $title,
                chunk.text = $text,
                chunk.search_text = $search_text,
                chunk.embedding = $embedding
            WITH chunk
            UNWIND $reason_codes AS code
            MERGE (reason:ReasonCode {code: code})
            SET reason.description = $reason_descriptions[code]
            MERGE (chunk)-[:EXPLAINS]->(reason)
            """,
            chunk_id=item["chunk_id"],
            title=item["title"],
            text=item["text"],
            search_text=search_text,
            embedding=embedding,
            reason_codes=item["reason_codes"],
            reason_descriptions=REASON_DESCRIPTIONS,
            database_=NEO4J_DATABASE,
        )


def load_payments(driver: Any) -> None:
    for item in load_json("payments.json"):
        driver.execute_query(
            """
            MERGE (customer:Customer {customer_id: $customer_id})
            MERGE (merchant:Merchant {merchant_id: $merchant_id})
              SET merchant.name = $merchant_name
            MERGE (acquirer:Acquirer {acquirer_id: $acquirer_id})
              SET acquirer.name = $acquirer_name
            MERGE (method:PaymentMethod {method_id: $method_id})
              SET method.brand = $brand, method.last4 = $last4
            MERGE (payment:Payment {payment_id: $payment_id})
              SET payment.amount = $amount,
                  payment.currency = $currency,
                  payment.status = $status,
                  payment.created_at = datetime($created_at)
            MERGE (customer)-[:INITIATED]->(payment)
            MERGE (payment)-[:AT_MERCHANT]->(merchant)
            MERGE (payment)-[:ROUTED_TO]->(acquirer)
            MERGE (payment)-[:USES]->(method)
            FOREACH (_ IN CASE WHEN $reason_code IS NULL THEN [] ELSE [1] END |
              MERGE (reason:ReasonCode {code: $reason_code})
              ON CREATE SET reason.description = $reason_description
              MERGE (payment)-[:FAILED_WITH]->(reason)
            )
            """,
            **item,
            reason_description=REASON_DESCRIPTIONS.get(item.get("reason_code") or "", ""),
            database_=NEO4J_DATABASE,
        )


def print_summary(driver: Any) -> None:
    records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WITH labels(n)[0] AS label, count(*) AS count
        RETURN label, count ORDER BY label
        """,
        database_=NEO4J_DATABASE,
    )
    print("Dataset GraphRAG cargado:")
    for record in records:
        print(f"  - {record['label']}: {record['count']}")


def main() -> None:
    embedder = SentenceTransformerEmbeddings(model=EMBEDDING_MODEL)
    probe = embedder.embed_query("dimension probe")
    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        wait_for_neo4j(driver)
        create_schema(driver, len(probe))
        load_knowledge(driver, embedder)
        load_payments(driver)
        wait_for_indexes(driver, [VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME])
        print_summary(driver)


if __name__ == "__main__":
    main()
