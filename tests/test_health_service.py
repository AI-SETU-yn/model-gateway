import pytest

from app.config.settings import ModelGatewayConfig
from app.services.health_service import HealthService


class StubProvider:
    async def complete(self, request):
        return type('Resp', (), {'content': 'ok'})()


@pytest.mark.asyncio
async def test_health_service_returns_up_status() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'qwen-erp',
                    'provider': 'openai',
                    'base_url': 'http://localhost:8000/v1',
                    'api_key': 'local',
                    'adapter_enabled': True,
                    'default_adapter': 'academic',
                }
            }
        }
    )
    response = await HealthService(StubProvider()).check_all(config.models)
    assert response.models[0].status == 'UP'
    assert response.models[0].adapter == 'academic'
