import asyncio

import pytest

from app.schemas.inference import GenerateRequest, GenerateResponse, UsageResponse
from app.services.generation.service import ResponseGenerationService


class StubInferenceService:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        return GenerateResponse(
            response=self.response,
            adapter=request.adapter,
            model='stub-model',
            usage=UsageResponse(promptTokens=1, completionTokens=1, totalTokens=2),
            generationTimeMs=1.0,
        )


@pytest.mark.parametrize(
    'response_type',
    ['enterprise', 'clarification', 'tool_failure', 'multi_tool', 'validation_retry', 'empty_result'],
)
def test_generate_routes_enterprise_response_types_to_enterprise_profile(response_type: str) -> None:
    enterprise = StubInferenceService('enterprise')
    general = StubInferenceService('general')
    service = ResponseGenerationService(enterprise, general)

    response = asyncio.run(service.generate(
        GenerateRequest.model_validate(
            {
                'adapter': 'service_alpha',
                'messages': [{'role': 'user', 'content': 'hello'}],
                'responseType': response_type,
            }
        )
    ))

    assert response.response == 'enterprise'
    assert len(enterprise.calls) == 1
    assert general.calls == []


@pytest.mark.parametrize('response_type', ['general', 'general_chat', 'current_info', 'planner_failure'])
def test_generate_routes_general_response_types_to_general_profile(response_type: str) -> None:
    enterprise = StubInferenceService('enterprise')
    general = StubInferenceService('general')
    service = ResponseGenerationService(enterprise, general)

    response = asyncio.run(service.generate(
        GenerateRequest.model_validate(
            {
                'messages': [{'role': 'user', 'content': 'hello'}],
                'responseType': response_type,
            }
        )
    ))

    assert response.response == 'general'
    assert enterprise.calls == []
    assert len(general.calls) == 1
