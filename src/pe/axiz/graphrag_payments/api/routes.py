from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j import Driver

from pe.axiz.graphrag_payments.application.generation import build_generator
from pe.axiz.graphrag_payments.application.service import GraphRagService
from pe.axiz.graphrag_payments.domain.models import (
    GraphRagQueryRequest,
    GraphRagQueryResponse,
    PaymentGraphResponse,
    SchemaResponse,
)
from pe.axiz.graphrag_payments.infrastructure.neo4j import check_connectivity, get_driver
from pe.axiz.graphrag_payments.settings import get_settings

router = APIRouter()


@lru_cache
def get_service() -> GraphRagService:
    settings = get_settings()
    driver = get_driver()
    return GraphRagService(
        driver=driver,
        settings=settings,
        retriever=None,
        generator=build_generator(settings),
    )


@router.get("/health/live", tags=["health"])
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", tags=["health"])
def health_ready(driver: Driver = Depends(get_driver)) -> dict[str, str]:
    try:
        check_connectivity(driver)
        return {"status": "ready", "neo4j": "reachable"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j no está disponible",
        ) from exc


@router.post(
    "/api/v1/graphrag/query",
    response_model=GraphRagQueryResponse,
    tags=["graphrag"],
)
def graphrag_query(
    request: GraphRagQueryRequest,
    service: GraphRagService = Depends(get_service),
) -> GraphRagQueryResponse:
    return service.query(request)


@router.get("/api/v1/graph/schema", response_model=SchemaResponse, tags=["graph"])
def graph_schema(service: GraphRagService = Depends(get_service)) -> SchemaResponse:
    return service.schema()


@router.get(
    "/api/v1/payments/{payment_id}/graph",
    response_model=PaymentGraphResponse,
    tags=["graph"],
)
def payment_graph(
    payment_id: str,
    service: GraphRagService = Depends(get_service),
) -> PaymentGraphResponse:
    result = service.payment_graph(payment_id)
    if not result.nodes:
        raise HTTPException(status_code=404, detail=f"Payment {payment_id} no encontrado")
    return result
