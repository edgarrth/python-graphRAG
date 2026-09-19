from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Axiz GraphRAG Payments PoC"
    environment: str = "local"

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "axiz-graphrag-poc"
    neo4j_database: str = "neo4j"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_index_name: str = "knowledge_embedding"
    fulltext_index_name: str = "knowledge_fulltext"
    default_top_k: int = Field(default=4, ge=1, le=20)
    max_top_k: int = Field(default=10, ge=1, le=50)

    generation_provider: Literal["deterministic", "openai"] = "deterministic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
