import pytest

from app.config.settings import GenerationConfig, ModelGatewayConfig
from app.services.planner import PlannerService


class StubInferenceService:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
        self.calls: list[tuple[str, str]] = []

    async def planner_generate(self, adapter: str, prompt: str):
        self.calls.append((adapter, prompt))
        return self.raw_response, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


@pytest.mark.asyncio
async def test_planner_service_maps_structured_lora_fields_into_response() -> None:
    config = ModelGatewayConfig(
        base_model='Qwen/Qwen2.5-1.5B-Instruct',
        default_adapter='academic',
        adapters_root='adapters',
        planner_system_prompt='You are the ERP Planner model for YN Setu.',
        generation=GenerationConfig(),
    )
    raw_response = (
        '{"confidence":0.98,"domain":"vidhya","entity":"academic_year","operation":"list",'
        '"service":"academic","requiresTool":true,"responseType":"structured",'
        '"parameters":{"academicYear":null}}'
    )
    service = PlannerService(config, StubInferenceService(raw_response))

    response = await service.plan(type('Request', (), {'adapter': 'academic', 'query': 'List all academic years'})())

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
