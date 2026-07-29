from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.health import HealthResponse, MetricsResponse
from app.schemas.inference import ModelHealthResponse
from app.services.dependencies import get_health_service, get_metrics_service, get_model_registry
from app.services.health_service import HealthService
from app.services.metrics_service import MetricsService

router = APIRouter(tags=['management'])


@router.get('/health', response_model=HealthResponse)
async def health(registry=Depends(get_model_registry)) -> HealthResponse:
    _, erp_profile = registry.planner_profile()
    return HealthResponse(
        status='ok',
        baseModelLoaded=True,
        defaultAdapter=erp_profile.default_adapter or '',
        loadedAdapters=[erp_profile.default_adapter] if erp_profile.default_adapter else [],
        device='litellm',
        dtype=erp_profile.provider,
    )


@router.get('/health/models', response_model=ModelHealthResponse)
async def health_models(registry=Depends(get_model_registry), service: HealthService = Depends(get_health_service)) -> ModelHealthResponse:
    return await service.check_all(registry.all_profiles())


@router.get('/metrics', response_model=MetricsResponse)
async def metrics(service: MetricsService = Depends(get_metrics_service)) -> MetricsResponse:
    return service.snapshot()
