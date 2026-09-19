from pe.axiz.graphrag_payments.infrastructure.schema import build_schema_statements


def test_vector_index_ddl_has_balanced_options_block() -> None:
    statements = build_schema_statements(
        vector_index_name="knowledge_embedding",
        fulltext_index_name="knowledge_fulltext",
        embedding_dimension=384,
    )

    vector_ddl = next(statement for statement in statements if "CREATE VECTOR INDEX" in statement)

    assert vector_ddl == (
        "CREATE VECTOR INDEX knowledge_embedding IF NOT EXISTS "
        "FOR (n:KnowledgeChunk) ON n.embedding "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, "
        "`vector.similarity_function`: 'cosine'}}"
    )
    assert vector_ddl.count("{") == vector_ddl.count("}")
