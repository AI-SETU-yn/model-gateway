from types import SimpleNamespace

import pytest

from app.clients.litellm_client import LiteLLMClient
from app.config.settings import ModelGatewayConfig, Settings
from app.models.inference import InferenceRequest


@pytest.mark.asyncio
async def test_litellm_client_builds_vllm_request(monkeypatch) -> None:
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='hello'), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16),
            model='qwen-erp',
        )

    monkeypatch.setattr('app.clients.litellm_client._get_litellm', lambda: SimpleNamespace(acompletion=fake_acompletion))

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
                }
            }
        }
    )
    client = LiteLLMClient(Settings(), config)
    response = await client.completion(
        InferenceRequest(
            modelAlias='erp',
            systemPrompt='system',
            userPrompt='user',
            temperature=0.1,
            top_p=0.9,
            maxTokens=32,
            metadata={'request_id': 'req-1', 'trace_id': 'trace-1'},
        )
    )

    assert captured['custom_llm_provider'] == 'openai'
    assert captured['api_base'] == 'http://localhost:4000/v1'
    assert captured['api_key'] == 'local'
    assert captured['model'] == 'qwen-erp'
    assert response.content == 'hello'
    assert response.usage.total_tokens == 16


@pytest.mark.asyncio
async def test_litellm_client_streams_incremental_chunks(monkeypatch) -> None:
    captured = {}

    class FakeStream:
        def __aiter__(self):
            self._chunks = iter([
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content='hel'), finish_reason=None)],
                    model='qwen-general',
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content='lo'), finish_reason='stop')],
                    model='qwen-general',
                    usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2, total_tokens=4),
                ),
            ])
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr('app.clients.litellm_client._get_litellm', lambda: SimpleNamespace(acompletion=fake_acompletion))

    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'general': {
                    'model_name': 'qwen-general',
                    'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'provider': {
                        'type': 'vllm',
                        'api_base': 'http://localhost:4000/v1',
                        'api_key': 'local',
                        'deployment_name': 'qwen-general',
                    },
                }
            }
        }
    )
    client = LiteLLMClient(Settings(), config)

    chunks = []
    async for chunk in client.stream_completion(
        InferenceRequest(
            modelAlias='general',
            systemPrompt='system',
            userPrompt='user',
            temperature=0.1,
            top_p=0.9,
            maxTokens=32,
            stream=True,
            metadata={'request_id': 'req-2', 'trace_id': 'trace-2'},
        )
    ):
        chunks.append(chunk)

    assert captured['stream'] is True
    assert [chunk.content for chunk in chunks] == ['hel', 'lo']
    assert chunks[-1].finish_reason == 'stop'
    assert chunks[-1].usage.total_tokens == 4
