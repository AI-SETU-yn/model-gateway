from app.main import create_app
from app.services.dependencies import get_health_service, get_metrics_service
from fastapi.testclient import TestClient


class StubMetricsService:
    def snapshot(self):
        return {
            'requestsTotal': 0,
            'generationRequests': 0,
            'plannerRequests': 0,
            'adapterReloadRequests': 0,
            'adapterCacheHits': 0,
            'adapterCacheMisses': 0,
            'failuresTotal': 0,
            'avgGenerationTimeMs': 0.0,
            'lastGenerationTimeMs': 0.0,
            'gpuMemoryAllocatedMb': 0.0,
            'processMemoryRssMb': 0.0,
        }


class StubHealthService:
    def __init__(self, ready: bool) -> None:
        self._ready = ready

    async def readiness(self, profiles) -> bool:
        return self._ready


def test_health_reports_remote_provider_shape() -> None:
    app = create_app()
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    with TestClient(app) as client:
        response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['baseModelLoaded'] is True
    assert body['defaultAdapter'] == 'academic'
    assert body['loadedAdapters'] == ['academic']
    assert body['device'] == 'remote'
    assert body['dtype'] == 'remote'


def test_ready_reports_not_ready_when_remote_inference_is_down() -> None:
    app = create_app()
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    app.dependency_overrides[get_health_service] = lambda: StubHealthService(False)
    with TestClient(app) as client:
        response = client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'not_ready'
    assert body['baseModelLoaded'] is False


def test_ready_reports_ok_when_remote_inference_is_up() -> None:
    app = create_app()
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    app.dependency_overrides[get_health_service] = lambda: StubHealthService(True)
    with TestClient(app) as client:
        response = client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['baseModelLoaded'] is True


def test_metrics_endpoint_contract_is_preserved() -> None:
    app = create_app()
    app.dependency_overrides[get_metrics_service] = lambda: StubMetricsService()
    with TestClient(app) as client:
        response = client.get('/metrics')
    assert response.status_code == 200
    body = response.json()
    assert body['requestsTotal'] == 0
    assert body['plannerRequests'] == 0
