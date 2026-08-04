import pytest

from app.config.settings import ModelGatewayConfig
from app.models.inference import InferenceResponse, InferenceUsage
from app.services.health_service import HealthService


class StubProvider:
    async def complete(self, request):
        return InferenceResponse(
            content='ok',
            finishReason='stop',
            usage=InferenceUsage(promptTokens=1, completionTokens=1, totalTokens=2),
            latencyMs=1.0,
            provider='vllm',
            model='qwen-erp',
            adapter=request.adapter,
            requestId='health',
            traceId='health',
            stream=False,
        )


@pytest.mark.asyncio
async def test_health_service_returns_up_status() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'qwen-erp',
                    'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'provider': {
                        'type': 'vllm',
                        'api_base': 'http://localhost:4000/v1',
                        'api_key': 'local',
                        'deployment_name': 'qwen-erp',
                    },
                    'adapter_enabled': True,
                    'default_adapter': 'academic',
                }
            }
        }
    )
    response = await HealthService(StubProvider()).check_all(config.models)
    assert response.models[0].status == 'UP'
    assert response.models[0].adapter == 'academic'
    assert response.models[0].provider == 'vllm'
    assert response.models[0].base_url == 'http://localhost:4000/v1'
