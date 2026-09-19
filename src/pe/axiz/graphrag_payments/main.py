from __future__ import annotations

from fastapi import FastAPI

from pe.axiz.graphrag_payments.api.routes import router
from pe.axiz.graphrag_payments.settings import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "PoC de GraphRAG para investigación operativa de fallas de payment processing. "
        "Combina recuperación híbrida con expansión de relaciones en Neo4j."
    ),
)
app.include_router(router)
