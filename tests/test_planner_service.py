import asyncio

from app.config.settings import GenerationConfig, ModelGatewayConfig
from app.services.planner import PlannerService


class StubInferenceService:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[tuple[str, str]] = []

    async def planner_generate(self, adapter: str, prompt: str):
        self.calls.append((adapter, prompt))
        return self.raw_response, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def _build_service(raw_response: str) -> PlannerService:
    return PlannerService(_build_config(), StubInferenceService(raw_response))


def _build_config() -> ModelGatewayConfig:
    return ModelGatewayConfig(
        base_model='Qwen/Qwen2.5-1.5B-Instruct',
        default_adapter='academic',
        adapters_root='adapters',
        planner_system_prompt='You are the ERP Planner model for YN Setu.',
        generation=GenerationConfig(),
    )


def _plan(raw_response: str, *, adapter: str = 'academic', query: str = 'List holidays'):
    service = _build_service(raw_response)
    return asyncio.run(service.plan(type('Request', (), {'adapter': adapter, 'query': query})()))


def test_planner_service_maps_structured_lora_fields_into_response() -> None:
    raw_response = (
        '{"confidence":0.98,"domain":"vidhya","entity":"academic_year","operation":"list",'
        '"service":"academic","requiresTool":true,"responseType":"structured",'
        '"parameters":{"academicYear":null}}'
    )

    response = _plan(raw_response, query='List all academic years')

    assert response.intent == 'academic.academic_year.list'
    assert response.tool is None
    assert response.domain == 'vidhya'
    assert response.service == 'academic'
    assert response.entity == 'academic_year'
    assert response.operation == 'list'
    assert response.parameters == {}
    assert response.requires_tool is True
    assert response.response_type == 'structured'
    assert response.confidence == 0.98
    assert response.raw_response == raw_response


def test_planner_service_preserves_clean_json_with_existing_intent_and_service() -> None:
    raw_response = (
        '{"intent":"academic.holiday.list","domain":"vidhya","service":"academic",'
        '"entity":"holiday","operation":"list","requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'academic.holiday.list'
    assert response.service == 'academic'
    assert response.entity == 'holiday'
    assert response.operation == 'list'
    assert response.requires_tool is True


def test_planner_service_ignores_trailing_text_after_json_object() -> None:
    raw_response = (
        '{"domain":"vidhya","service":"academic","entity":"holiday","operation":"list",'
        '"requiresTool":true,"parameters":{}}\n'
        'Human: Can you please tell me'
    )

    response = _plan(raw_response)

    assert response.intent == 'academic.holiday.list'
    assert response.service == 'academic'
    assert response.entity == 'holiday'
    assert response.operation == 'list'
    assert response.requires_tool is True


def test_planner_service_derives_intent_when_missing() -> None:
    raw_response = (
        '{"domain":"vidhya","service":"academic","entity":"holiday","operation":"list",'
        '"requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'academic.holiday.list'
    assert response.service == 'academic'


def test_planner_service_derives_missing_service_from_adapter() -> None:
    raw_response = (
        '{"domain":"vidhya","entity":"holiday","operation":"list",'
        '"requiresTool":true,"parameters":{}}'
    )

    response = _plan(raw_response)

    assert response.intent == 'academic.holiday.list'
    assert response.service == 'academic'
    assert response.entity == 'holiday'
    assert response.operation == 'list'


def test_planner_service_ignores_malformed_trailing_conversational_output() -> None:
    raw_response = (
        '{"domain":"vidhya","entity":"holiday","operation":"list",'
        '"requiresTool":true,"parameters":{}}\n'
        'Human: Can you please tell me {"this is not": valid json'
    )

    response = _plan(raw_response)

    assert response.intent == 'academic.holiday.list'
    assert response.service == 'academic'
    assert response.requires_tool is True
