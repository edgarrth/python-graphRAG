from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from neo4j import Driver
from neo4j.exceptions import Neo4jError

from pe.axiz.graphrag_payments.application.benchmark import (
    BenchmarkNotReadyError,
    BenchmarkService,
)
from pe.axiz.graphrag_payments.application.graphsage import (
    GraphSageNotReadyError,
    GraphSagePaymentNotFoundError,
    GraphSageService,
)
from pe.axiz.graphrag_payments.application.generation import build_generator
from pe.axiz.graphrag_payments.application.service import GraphRagService
from pe.axiz.graphrag_payments.domain.models import (
    BenchmarkResponse,
    GraphRagQueryRequest,
    GraphRagQueryResponse,
    GraphSageSimilarResponse,
    GraphSageStatus,
    GraphSageTrainRequest,
    GraphSageTrainResponse,
    PaymentGraphResponse,
    SchemaResponse,
)
from pe.axiz.graphrag_payments.infrastructure.neo4j import check_connectivity, get_driver
from pe.axiz.graphrag_payments.settings import get_settings

router = APIRouter()
LOGGER = logging.getLogger(__name__)


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


def _encode_sse(event: dict[str, object]) -> str:
    event_name = str(event.get("event", "message"))
    payload = json.dumps(event.get("data", {}), ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n"


@router.get("/health/live", tags=["health"])
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", tags=["health"])
def health_ready(driver: Driver = Depends(get_driver)) -> dict[str, str | bool]:
    settings = get_settings()
    try:
        check_connectivity(driver)
        return {
            "status": "ready",
            "neo4j": "reachable",
            "generation_provider": settings.generation_provider,
            "generation_model": (
                settings.openai_model if settings.generation_provider == "openai" else "deterministic"
            ),
            "openai_key_configured": bool(settings.openai_api_key),
        }
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


@router.post("/api/v1/graphrag/query/stream", tags=["graphrag"])
def graphrag_query_stream(
    request: GraphRagQueryRequest,
    service: GraphRagService = Depends(get_service),
) -> StreamingResponse:
    def event_source() -> Iterator[str]:
        try:
            for event in service.query_stream(request):
                yield _encode_sse(event)
        except Exception as exc:
            yield _encode_sse(
                {
                    "event": "error",
                    "data": {"message": "La consulta GraphRAG falló.", "detail": str(exc)},
                }
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@router.get("/api/v1/graphsage/status", response_model=GraphSageStatus, tags=["graphsage"])
def graphsage_status(driver: Driver = Depends(get_driver)) -> GraphSageStatus:
    return GraphSageService(driver, get_settings()).status()


@router.post("/api/v1/graphsage/train", response_model=GraphSageTrainResponse, tags=["graphsage"])
def graphsage_train(
    request: GraphSageTrainRequest,
    driver: Driver = Depends(get_driver),
) -> GraphSageTrainResponse:
    try:
        return GraphSageService(driver, get_settings()).train(request)
    except GraphSageNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/api/v1/graphsage/payments/{payment_id}/similar",
    response_model=GraphSageSimilarResponse,
    tags=["graphsage"],
)
def graphsage_similar(
    payment_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
    min_similarity: float = Query(default=0.0, ge=-1.0, le=1.0),
    driver: Driver = Depends(get_driver),
) -> GraphSageSimilarResponse:
    try:
        return GraphSageService(driver, get_settings()).similar(
            payment_id, top_k=top_k, min_similarity=min_similarity
        )
    except GraphSagePaymentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pago {payment_id} no encontrado") from exc
    except GraphSageNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/v1/experiments/graphsage-value", response_model=BenchmarkResponse, tags=["experiments"])
def graphsage_value_benchmark(
    top_k: int = Query(default=5, ge=1, le=20),
    driver: Driver = Depends(get_driver),
) -> BenchmarkResponse:
    try:
        return BenchmarkService(driver, get_settings().neo4j_database).run(top_k=top_k)
    except BenchmarkNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Neo4jError as exc:
        LOGGER.exception("Fallo de Neo4j en el experimento GraphSAGE")
        raise HTTPException(
            status_code=503,
            detail="Falló la evaluación en Neo4j. Revisa los logs de la API.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Fallo inesperado en el experimento GraphSAGE")
        raise HTTPException(
            status_code=500,
            detail="Error de evaluación. Revisa los logs de la API.",
        ) from exc
