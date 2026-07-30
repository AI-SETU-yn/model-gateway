from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.health import HealthResponse, MetricsResponse
from app.schemas.inference import ModelHealthResponse
from app.services.dependencies import get_erp_loader, get_health_service, get_metrics_service, get_model_registry
from app.services.health_service import HealthService
from app.services.metrics_service import MetricsService

router = APIRouter(tags=['management'])


@router.get('/health', response_model=HealthResponse)
async def health(registry=Depends(get_model_registry), loader=Depends(get_erp_loader)) -> HealthResponse:
    _, erp_profile = registry.planner_profile()
    return HealthResponse(
        status='ok',
        baseModelLoaded=loader.base_model_loaded,
        defaultAdapter=erp_profile.default_adapter or '',
        loadedAdapters=loader.loaded_adapters,
        device=loader.resolved_device,
        dtype=loader.resolved_dtype,
    )


@router.get('/health/models', response_model=ModelHealthResponse)
async def health_models(registry=Depends(get_model_registry), service: HealthService = Depends(get_health_service)) -> ModelHealthResponse:
    return await service.check_all(registry.all_profiles())


@router.get('/metrics', response_model=MetricsResponse)
async def metrics(service: MetricsService = Depends(get_metrics_service)) -> MetricsResponse:
    return service.snapshot()
