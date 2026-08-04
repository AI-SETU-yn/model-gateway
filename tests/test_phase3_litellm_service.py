import httpx
import pytest

from app.core.config import Settings
from app.core.model_registry import ModelRegistry
from app.services.litellm_service import LiteLLMService


class StubAsyncClient:
    def __init__(self, payload=None, status_code=200, text='ok'):
        self.payload = payload or {'data': [{'id': 'qwen-erp'}]}
        self.status_code = status_code
        self.text = text

    async def get(self, url, headers=None):
        request = httpx.Request('GET', url)
        return httpx.Response(self.status_code, json=self.payload, request=request, text=self.text)


@pytest.mark.asyncio
async def test_check_connectivity_reports_healthy(monkeypatch, tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text(
        'models:\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4000/v1\n    supports_streaming: true\n    default: true\n',
        encoding='utf-8',
    )
    settings = Settings.model_validate(
        {
            'LITELLM_BASE_URL': 'http://127.0.0.1:4000/v1',
            'DEFAULT_MODEL': 'qwen-erp',
            'REQUEST_TIMEOUT': 60,
            'MAX_RETRIES': 3,
            'MODEL_REGISTRY_PATH': str(registry_path),
        }
    )
    service = LiteLLMService(settings, ModelRegistry(registry_path), StubAsyncClient())

    async def fake_generate_chat(request, *, model, profile):
        return {'id': 'chatcmpl-1', 'model': model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'pong'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    monkeypatch.setattr(service, 'generate_chat', fake_generate_chat)
    monkeypatch.setattr(service, '_get_litellm', staticmethod(lambda: object()))

    result = await service.check_connectivity()

    assert result.healthy is True
    assert result.checks['litellm_sdk_available'] is True
    assert result.checks['upstream_reachable'] is True
    assert result.checks['configured_model_remote_available'] is True


def test_error_mapping_covers_standard_gateway_statuses():
    request = httpx.Request('GET', 'http://localhost')
    response = httpx.Response(429, request=request, text='rate limited')
    exc = httpx.HTTPStatusError('rate limited', request=request, response=response)

    mapped = LiteLLMService._map_exception(exc)

    assert mapped.status_code == 429
    assert mapped.code == 'RATE_LIMITED'
