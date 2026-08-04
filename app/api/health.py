"""Health API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_litellm_service, get_settings
from app.middleware.request_id import request_id_var, span_id_var, trace_id_var
from app.models.response import HealthCheckResponse
from app.services.litellm_service import LiteLLMService

router = APIRouter(prefix='/health', tags=['health'])


@router.get('/live', response_model=HealthCheckResponse)
async def liveness(settings: Settings = Depends(get_settings)) -> HealthCheckResponse:
    """Verify process health, configuration, and dependency wiring."""

    return HealthCheckResponse(
        status='ok',
        service=settings.app_name,
        checks={
            'process': True,
            'configuration': True,
            'dependency_injection': True,
            'environment': settings.environment,
        },
        request_id=request_id_var.get() or None,
        trace_id=trace_id_var.get() or None,
        span_id=span_id_var.get() or None,
    )


@router.get('/ready', response_model=HealthCheckResponse)
async def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
    service: LiteLLMService = Depends(get_litellm_service),
) -> HealthCheckResponse:
    """Verify LiteLLM reachability, remote model exposure, and inference readiness."""

    result = await service.check_connectivity()
    if not result.healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthCheckResponse(
        status='ready' if result.healthy else 'not_ready',
        service=settings.app_name,
        checks=result.checks,
        request_id=request_id_var.get() or None,
        trace_id=trace_id_var.get() or None,
        span_id=span_id_var.get() or None,
    )


@router.get('', response_model=HealthCheckResponse)
async def health_summary(
    response: Response,
    settings: Settings = Depends(get_settings),
    service: LiteLLMService = Depends(get_litellm_service),
) -> HealthCheckResponse:
    """Convenience endpoint mirroring readiness for orchestration checks."""

    return await readiness(response=response, settings=settings, service=service)
