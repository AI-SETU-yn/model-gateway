import app.api.routers.management as management_module
import app.main as main_module
from app.main import create_app
from app.services.dependencies import get_erp_model_loader, get_metrics_service, get_model_registry
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
        if preload_default_adapter and 'service_alpha' not in self.loaded_adapters:
            self.loaded_adapters.append('service_alpha')

    async def reload_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.loaded_adapters:
            self.loaded_adapters.append(adapter_name)


class StubGeneralModelLoader:
    def __init__(self) -> None:
        self.base_model_loaded = True
        self.initialize_calls = 0

    async def initialize(self, preload_default_adapter: bool = False) -> None:
        self.initialize_calls += 1


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


class StubRegistry:
    def planner_profile(self):
        return 'erp', type('Profile', (), {'default_adapter': None})()

    def all_profiles(self):
        return {}


class StubHealthService:
    def __init__(self) -> None:
        self.calls = 0

    async def check_all(self, profiles):
        self.calls += 1
        return type('ModelHealthResponse', (), {'models': []})()


def _build_client(stub_loader: StubModelLoader) -> TestClient:
    main_module.get_erp_model_loader = lambda: stub_loader
    main_module.get_general_model_loader = lambda: StubGeneralModelLoader()
    main_module.get_model_registry = lambda: StubRegistry()
    main_module.get_health_service = lambda: StubHealthService()
    app = create_app()
    app.dependency_overrides[get_erp_model_loader] = lambda: stub_loader
    app.dependency_overrides[get_model_registry] = lambda: StubRegistry()
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    return TestClient(app)


def test_startup_does_not_preload_business_default_adapter() -> None:
    stub_loader = StubModelLoader()
    with _build_client(stub_loader):
        pass
    assert stub_loader.initialize_calls == []
    assert stub_loader.loaded_adapters == []


def test_health_reports_configured_default_adapter() -> None:
    stub_loader = StubModelLoader()
    with _build_client(stub_loader) as client:
        response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['baseModelLoaded'] is True
    assert body['defaultAdapter'] == ''
    assert body['loadedAdapters'] == []
    assert body['device'] == 'cuda'
    assert body['dtype'] == 'auto'


