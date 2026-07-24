import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse, UsageResponse
from app.providers import litellm_provider as litellm_provider_module
from app.providers.litellm_provider import LiteLLMProvider
from app.services.dependencies import get_gateway_service, get_provider
from app.services.gateway_service import GatewayService

os.environ['MODEL_GATEWAY_MODEL_NAME'] = 'qwen2.5:1.5b'


class StubProvider:
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            response='stub response',
            usage=UsageResponse(promptTokens=1, completionTokens=2, totalTokens=3),
            model='ollama/qwen2.5:1.5b',
        )


@pytest.fixture()
def client():
    app = create_app()
    app.dependency_overrides[get_provider] = lambda: StubProvider()
    app.dependency_overrides[get_gateway_service] = lambda: GatewayService(StubProvider())
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_ready(client: TestClient):
    response = client.get('/ready')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_generate_endpoint(client: TestClient):
    response = client.post(
        '/generate',
        json={
            'systemPrompt': 'You are helpful',
            'userPrompt': 'Hello',
            'conversationId': 'conv-1',
            'metadata': {'tenant': 't1'},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body['response'] == 'stub response'
    assert body['model'] == 'ollama/qwen2.5:1.5b'
    assert body['usage']['totalTokens'] == 3


@pytest.mark.asyncio
async def test_gateway_service():
    service = GatewayService(StubProvider())
    response = await service.generate(
        GenerateRequest(systemPrompt='sys', userPrompt='usr', conversationId='c1', metadata={})
    )
    assert response.response == 'stub response'


def test_litellm_provider_response_mapping():
    provider = LiteLLMProvider(
        type(
            'Settings',
            (),
            {
                'request_timeout_seconds': 60.0,
                'connect_timeout_seconds': 5.0,
                'read_timeout_seconds': 60.0,
                'max_retries': 0,
                'ollama_base_url': 'http://localhost:11434',
                'model_name': 'qwen2.5:1.5b',
            },
        )()
    )
    mapped = provider._to_response(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='hello'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model='ollama/qwen2.5:1.5b',
        )
    )
    assert mapped.response == 'hello'
    assert mapped.usage.total_tokens == 15
    assert mapped.model == 'ollama/qwen2.5:1.5b'


@pytest.mark.asyncio
async def test_litellm_provider_calls_sdk(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='sdk response'))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            model='ollama/qwen2.5:1.5b',
        )

    monkeypatch.setattr(litellm_provider_module.litellm, 'acompletion', fake_acompletion)

    provider = LiteLLMProvider(
        type(
            'Settings',
            (),
            {
                'request_timeout_seconds': 60.0,
                'connect_timeout_seconds': 5.0,
                'read_timeout_seconds': 60.0,
                'max_retries': 0,
                'ollama_base_url': 'http://localhost:11434',
                'model_name': 'qwen2.5:1.5b',
            },
        )()
    )
    response = await provider.generate(
        GenerateRequest(systemPrompt='system', userPrompt='user', conversationId='c1', metadata={'tenant': 't1'})
    )

    assert response.response == 'sdk response'
    assert captured['model'] == 'ollama/qwen2.5:1.5b'
    assert captured['api_base'] == 'http://localhost:11434'
    assert captured['messages'][0]['role'] == 'system'
    assert captured['messages'][1]['role'] == 'user'
