import asyncio

from app.config.settings import GenerationConfig, ModelGatewayConfig
from app.registry.prompt_registry import PromptRegistry
from app.services.planner import PlannerService


class StubInferenceService:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[tuple[str, str]] = []

    async def planner_generate(self, adapter: str, prompt: str):
        self.calls.append((adapter, prompt))
        return self.raw_response, 0.0, 'test-model'


class FailingInferenceService:
    async def planner_generate(self, adapter: str, prompt: str):
        raise RuntimeError('model unavailable')


def _build_config() -> ModelGatewayConfig:
    return ModelGatewayConfig(
        models={
            'erp': {
                'model_name': 'qwen-erp',
                'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                'provider': 'transformers',
                'adapter_enabled': True,
                'default_adapter': 'service_alpha',
                'adapters_root': 'adapters',
                'prompts': {'planner': 'You are the ERP Planner model for YN Setu.'},
                'generation': GenerationConfig().model_dump(),
            },
            'general': {
                'model_name': 'qwen-general',
                'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                'provider': 'transformers',
                'adapter_enabled': False,
            },
        },
        routing={
            'planner_model': 'erp',
            'generate_model': 'erp',
            'security_model': 'general',
            'general_chat_model': 'general',
        },
    )


def _build_service(raw_response: str) -> tuple[PlannerService, StubInferenceService]:
    profile = _build_config().get_profile('erp')
    inference = StubInferenceService(raw_response)
    return PlannerService('erp', profile, PromptRegistry(), inference), inference


def _plan(raw_response: str, *, adapter: str = 'service_alpha', query: str = 'List records', **extra):
    service, _ = _build_service(raw_response)
    payload = {'adapter': adapter, 'query': query, 'prompt': None, 'registry_context': None, 'registry_metadata': None}
    payload.update(extra)
    request = type('Request', (), payload)()
    return asyncio.run(service.plan(request))


def test_planner_service_maps_structured_model_fields_into_response() -> None:
    raw_response = (
        '{"confidence":0.98,"domain":"example","entity":"record_group","operation":"list",'
        '"service":"service_alpha","requiresTool":true,"responseType":"structured",'
        '"parameters":{"recordGroup":null}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'service_alpha.record_group.list'
    assert response.tool is None
    assert response.domain == 'example'
    assert response.service == 'service_alpha'
    assert response.entity == 'record_group'
    assert response.operation == 'list'
    assert response.parameters == {}
    assert response.requires_tool is True
    assert response.response_type == 'structured'
    assert response.confidence == 0.98
    assert response.raw_response == raw_response


def test_planner_service_preserves_clean_json_with_existing_intent_and_service() -> None:
    raw_response = (
        '{"intent":"service_alpha.record_group.list","domain":"example","service":"service_alpha",'
        '"entity":"record_group","operation":"list","requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'service_alpha.record_group.list'
    assert response.service == 'service_alpha'
    assert response.entity == 'record_group'
    assert response.operation == 'list'
    assert response.requires_tool is True


def test_planner_service_ignores_trailing_text_after_json_object() -> None:
    raw_response = (
        '{"domain":"example","service":"service_alpha","entity":"record_group","operation":"list",'
        '"requiresTool":true,"parameters":{}}\n'
        'Human: Can you please tell me'
    )

    response = _plan(raw_response)

    assert response.intent == 'service_alpha.record_group.list'
    assert response.service == 'service_alpha'
    assert response.entity == 'record_group'
    assert response.operation == 'list'
    assert response.requires_tool is True


def test_planner_service_derives_intent_when_missing() -> None:
    raw_response = (
        '{"domain":"example","service":"service_alpha","entity":"record_group","operation":"list",'
        '"requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'service_alpha.record_group.list'
    assert response.service == 'service_alpha'


def test_planner_service_derives_missing_service_from_adapter() -> None:
    raw_response = (
        '{"domain":"example","entity":"record_group","operation":"list",'
        '"requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'service_alpha.record_group.list'
    assert response.service == 'service_alpha'
    assert response.entity == 'record_group'
    assert response.operation == 'list'


def test_planner_service_preserves_execution_plan_contract() -> None:
    raw_response = (
        '{"intent":"service_alpha.final.list","requiresTool":true,'
        '"executionPlan":['
        '{"stepId":"step_1","domain":"example","service":"service_alpha","entity":"source","operation":"list"},'
        '{"stepId":"step_2","domain":"example","service":"service_alpha","entity":"final","operation":"list",'
        '"dependsOn":["step_1"],"parameterBindings":{"source_id":{"from_step":"step_1","path":"$.data.items[0].id"}}}'
        ']}'
    )

    response = _plan(raw_response)

    assert response.requires_tool is True
    assert len(response.execution_plan) == 2
    assert response.execution_plan[0].step_id == 'step_1'
    assert response.execution_plan[1].depends_on == ['step_1']
    assert response.execution_plan[1].parameter_bindings == {
        'source_id': {'from_step': 'step_1', 'path': '$.data.items[0].id'}
    }


def test_planner_service_uses_ai_runtime_supplied_prompt_verbatim() -> None:
    service, inference = _build_service('{"intent":"general.chat","requiresTool":false}')
    request = type(
        'Request',
        (),
        {
            'adapter': 'service_alpha',
            'query': 'List records',
            'prompt': 'registry-aware prompt from runtime',
            'registry_context': None,
            'registry_metadata': None,
        },
    )()

    asyncio.run(service.plan(request))

    assert inference.calls == [('service_alpha', 'registry-aware prompt from runtime')]


def test_planner_service_includes_ai_runtime_registry_context_when_supplied() -> None:
    service, inference = _build_service('{"intent":"general.chat","requiresTool":false}')
    request = type(
        'Request',
        (),
        {
            'adapter': 'service_alpha',
            'query': 'List records',
            'prompt': None,
            'registry_context': 'Registered target: example/service_alpha/record_group/list',
            'registry_metadata': {'tool_count': 1},
        },
    )()

    asyncio.run(service.plan(request))

    prompt = inference.calls[0][1]
    assert 'Registry context supplied by AI Runtime' in prompt
    assert 'Registered target: example/service_alpha/record_group/list' in prompt
    assert '"tool_count": 1' in prompt


def test_planner_service_falls_back_to_general_chat_when_inference_raises() -> None:
    profile = _build_config().get_profile('erp')
    service = PlannerService('erp', profile, PromptRegistry(), FailingInferenceService())

    response = asyncio.run(
        service.plan(type('Request', (), {'adapter': 'service_alpha', 'query': 'List records', 'prompt': None, 'registry_context': None, 'registry_metadata': None})())
    )

    assert response.intent == 'general.chat'
    assert response.domain is None
    assert response.service is None
    assert response.entity is None
    assert response.operation is None
    assert response.requires_tool is False
    assert response.raw_response == 'fallback:RuntimeError'


def test_planner_service_falls_back_to_general_chat_when_model_returns_invalid_json() -> None:
    response = _plan('not json')

    assert response.intent == 'general.chat'
    assert response.requires_tool is False
    assert response.raw_response == 'fallback:PlannerResponseError'
