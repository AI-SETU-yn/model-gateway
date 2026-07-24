import app.api.routers.management as management_module
import app.main as main_module
from app.main import create_app
from app.services.dependencies import get_metrics_service, get_model_loader
from fastapi.testclient import TestClient


class StubModelLoader:
    def __init__(self) -> None:
        self.base_model_loaded = True
        self.loaded_adapters: list[str] = []
        self.resolved_device = 'cuda'
        self.resolved_dtype = 'auto'
        self.initialize_calls: list[bool] = []

    async def initialize(self, preload_default_adapter: bool = False) -> None:
        self.initialize_calls.append(preload_default_adapter)
        if preload_default_adapter and 'academic' not in self.loaded_adapters:
            self.loaded_adapters.append('academic')

    async def reload_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)


class StubMetricsService:
    def __init__(self) -> None:
        self.reload_calls = 0

    def record_adapter_reload(self) -> None:
        self.reload_calls += 1

    def snapshot(self):
        return {
            'requestsTotal': 0,
            'generationRequests': 0,
            'plannerRequests': 0,
            'adapterReloadRequests': self.reload_calls,
            'adapterCacheHits': 0,
            'adapterCacheMisses': 0,
            'failuresTotal': 0,
            'avgGenerationTimeMs': 0.0,
            'lastGenerationTimeMs': 0.0,
            'gpuMemoryAllocatedMb': 0.0,
            'processMemoryRssMb': 0.0,
        }


def _build_client(stub_loader: StubModelLoader) -> TestClient:
    main_module.get_model_loader = lambda: stub_loader
    management_module.get_model_config = lambda: type('Config', (), {'default_adapter': 'academic'})()
    app = create_app()
    app.dependency_overrides[get_model_loader] = lambda: stub_loader
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    return TestClient(app)


def test_startup_preloads_configured_default_adapter() -> None:
    stub_loader = StubModelLoader()
    with _build_client(stub_loader):
        pass
    assert stub_loader.initialize_calls == [True]
    assert stub_loader.loaded_adapters == ['academic']


def test_health_reports_configured_default_adapter() -> None:
    stub_loader = StubModelLoader()
    with _build_client(stub_loader) as client:
        response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['baseModelLoaded'] is True
    assert body['defaultAdapter'] == 'academic'
    assert body['loadedAdapters'] == ['academic']
    assert body['device'] == 'cuda'
    assert body['dtype'] == 'auto'


def test_reload_adapter_keeps_management_path_working() -> None:
    stub_loader = StubModelLoader()
    with _build_client(stub_loader) as client:
        response = client.post('/reload-adapter', json={'adapter': 'hrms'})
    assert response.status_code == 200
    body = response.json()
    assert body['adapter'] == 'hrms'
    assert body['reloaded'] is True
    assert body['loadedAdapters'] == ['academic', 'hrms']
