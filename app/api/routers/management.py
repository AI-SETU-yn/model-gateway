from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.schemas.health import HealthResponse, MetricsResponse
from app.schemas.inference import ModelHealthResponse
from app.services.dependencies import get_health_service, get_metrics_service, get_model_registry
from app.services.health_service import HealthService
from app.services.metrics_service import MetricsService

router = APIRouter(tags=['management'])


@router.get('/health', response_model=HealthResponse)
async def health(registry=Depends(get_model_registry)) -> HealthResponse:
    _, erp_profile = registry.planner_profile()
    loaded_adapters = [erp_profile.default_adapter] if erp_profile.adapter_enabled and erp_profile.default_adapter else []
    return HealthResponse(
        status='ok',
        baseModelLoaded=True,
        defaultAdapter=erp_profile.default_adapter or '',
        loadedAdapters=loaded_adapters,
        device=erp_profile.device,
        dtype=erp_profile.dtype,
    )


@router.get('/ready', response_model=HealthResponse)
async def ready(request: Request, registry=Depends(get_model_registry), service: HealthService = Depends(get_health_service)) -> HealthResponse:
    _, erp_profile = registry.planner_profile()
    loaded_adapters = [erp_profile.default_adapter] if erp_profile.adapter_enabled and erp_profile.default_adapter else []
    is_ready = await service.readiness(registry.all_profiles())
    request.app.state.is_ready = is_ready
    return HealthResponse(
        status='ok' if is_ready else 'not_ready',
        baseModelLoaded=is_ready,
        defaultAdapter=erp_profile.default_adapter or '',
        loadedAdapters=loaded_adapters,
        device=erp_profile.device,
        dtype=erp_profile.dtype,
    )


@router.get('/health/models', response_model=ModelHealthResponse)
async def health_models(registry=Depends(get_model_registry), service: HealthService = Depends(get_health_service)) -> ModelHealthResponse:
    return await service.check_all(registry.all_profiles())


@router.get('/metrics', response_model=MetricsResponse)
async def metrics(service: MetricsService = Depends(get_metrics_service)) -> MetricsResponse:
    return service.snapshot()
