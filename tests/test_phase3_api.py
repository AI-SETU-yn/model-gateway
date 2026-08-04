import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_chat_service, get_litellm_service, get_settings
from app.core.exceptions import register_exception_handlers
from app.models.request import ChatRequest
from app.models.response import ChatCompletionResponse, HealthCheckResponse, Usage
from app.services.litellm_service import ReadinessResult


class StubChatService:
    async def list_models(self):
        return {'object': 'list', 'data': [{'id': 'qwen-erp', 'object': 'model', 'owned_by': 'litellm:vllm'}]}

    async def generate_chat(self, request: ChatRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id='chatcmpl-1',
            model=request.model or 'qwen-erp',
            choices=[{'index': 0, 'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(self, request: ChatRequest):
        async def event_stream():
            yield 'data: {"id": "chunk-1"}\n\n'
            yield 'data: [DONE]\n\n'

        return event_stream()


class StubLiteLLMService:
    def __init__(self, healthy=True):
        self.healthy = healthy

    async def check_connectivity(self):
        return ReadinessResult(
            healthy=self.healthy,
            checks={
                'litellm_sdk_available': True,
                'upstream_reachable': self.healthy,
                'configured_model_registered': True,
                'configured_model_remote_available': self.healthy,
                'remote_inference_ready': self.healthy,
                'model_registry_loaded': True,
            },
        )


class StubSettings:
    app_name = 'yn-ai-setu-model-gateway'
    environment = 'test'


def build_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(health_router)
    return app


def test_chat_endpoint_returns_standard_response():
    app = build_test_app()
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()

    with TestClient(app) as client:
        response = client.post('/chat', json={'messages': [{'role': 'user', 'content': 'hello'}]})

    assert response.status_code == 200
    body = response.json()
    assert body['model'] == 'qwen-erp'
    assert body['choices'][0]['message']['content'] == 'hello'


def test_chat_endpoint_streams_sse_response():
    app = build_test_app()
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()

    with TestClient(app) as client:
        response = client.post('/chat', json={'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True})

    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    assert 'data: {"id": "chunk-1"}' in response.text
    assert 'data: [DONE]' in response.text


def test_ready_endpoint_returns_503_when_not_ready():
    app = build_test_app()
    app.dependency_overrides[get_settings] = lambda: StubSettings()
    app.dependency_overrides[get_litellm_service] = lambda: StubLiteLLMService(healthy=False)

    with TestClient(app) as client:
        response = client.get('/health/ready')

    assert response.status_code == 503
    body = response.json()
    assert body['status'] == 'not_ready'
    assert body['checks']['upstream_reachable'] is False


def test_ready_endpoint_returns_detailed_checks_when_ready():
    app = build_test_app()
    app.dependency_overrides[get_settings] = lambda: StubSettings()
    app.dependency_overrides[get_litellm_service] = lambda: StubLiteLLMService(healthy=True)

    with TestClient(app) as client:
        response = client.get('/health/ready')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert body['checks']['litellm_sdk_available'] is True
    assert body['checks']['configured_model_remote_available'] is True
