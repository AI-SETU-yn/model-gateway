from fastapi import APIRouter, Depends

from app.config.settings import get_model_config
from app.schemas.health import AdapterInventoryResponse, HealthResponse, MetricsResponse, ReloadAdapterResponse
from app.schemas.inference import ReloadAdapterRequest
from app.services.dependencies import get_adapter_manager, get_metrics_service, get_model_loader
from app.services.metrics_service import MetricsService
from app.services.model_loader import ModelLoader

router = APIRouter(tags=['management'])


@router.get('/health', response_model=HealthResponse)
async def health(model_loader: ModelLoader = Depends(get_model_loader)) -> HealthResponse:
    config = get_model_config()
    return HealthResponse(
        status='ok' if model_loader.base_model_loaded else 'starting',
        baseModelLoaded=model_loader.base_model_loaded,
        defaultAdapter=config.default_adapter,
        loadedAdapters=model_loader.loaded_adapters,
        device=model_loader.resolved_device,
        dtype=model_loader.resolved_dtype,
    )


@router.get('/metrics', response_model=MetricsResponse)
async def metrics(service: MetricsService = Depends(get_metrics_service)) -> MetricsResponse:
    return service.snapshot()


@router.get('/adapters', response_model=AdapterInventoryResponse)
async def list_adapters(adapter_manager=Depends(get_adapter_manager)) -> AdapterInventoryResponse:
    return AdapterInventoryResponse(adapters=adapter_manager.list_adapters())


@router.post('/reload-adapter', response_model=ReloadAdapterResponse)
async def reload_adapter(payload: ReloadAdapterRequest, loader: ModelLoader = Depends(get_model_loader), service: MetricsService = Depends(get_metrics_service)) -> ReloadAdapterResponse:
    await loader.reload_adapter(payload.adapter)
    service.record_adapter_reload()
    return ReloadAdapterResponse(adapter=payload.adapter, reloaded=True, loadedAdapters=loader.loaded_adapters)
