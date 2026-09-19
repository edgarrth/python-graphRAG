from __future__ import annotations


def build_schema_statements(
    *,
    vector_index_name: str,
    fulltext_index_name: str,
    embedding_dimension: int,
) -> list[str]:
    """Return idempotent Neo4j schema DDL required by the PoC.

    Keeping DDL construction in a pure function makes the generated Cypher
    independently testable without requiring a running Neo4j instance.
    """
    return [
        "CREATE CONSTRAINT payment_id IF NOT EXISTS FOR (n:Payment) REQUIRE n.payment_id IS UNIQUE",
        "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE",
        "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (n:Merchant) REQUIRE n.merchant_id IS UNIQUE",
        "CREATE CONSTRAINT acquirer_id IF NOT EXISTS FOR (n:Acquirer) REQUIRE n.acquirer_id IS UNIQUE",
        "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (n:PaymentMethod) REQUIRE n.method_id IS UNIQUE",
        "CREATE CONSTRAINT reason_code IF NOT EXISTS FOR (n:ReasonCode) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:KnowledgeChunk) REQUIRE n.chunk_id IS UNIQUE",
        (
            f"CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS "
            "FOR (n:KnowledgeChunk) ON n.embedding "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {embedding_dimension}, "
            "`vector.similarity_function`: 'cosine'}}"
        ),
        (
            f"CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS "
            "FOR (n:KnowledgeChunk) ON EACH [n.search_text]"
        ),
    ]
