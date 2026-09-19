from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from pe.axiz.graphrag_payments.api.routes import router
from pe.axiz.graphrag_payments.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    logging.getLogger("uvicorn.error").info(
        "GraphRAG runtime: generation_provider=%s model=%s openai_key_configured=%s",
        settings.generation_provider,
        settings.openai_model if settings.generation_provider == "openai" else "deterministic",
        bool(settings.openai_api_key),
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.4.0",
    description=(
        "PoC de GraphRAG para investigación operativa de fallas de payment processing. "
        "Combina recuperación híbrida con expansión de relaciones en Neo4j y streaming SSE."
    ),
    lifespan=lifespan,
)
app.include_router(router)
