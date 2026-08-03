"""Response generation service for ERP model flows."""

from __future__ import annotations

from app.schemas.inference import GenerateRequest, GenerateResponse
from app.services.inference import ERPInferenceService, GeneralInferenceService


class ResponseGenerationService:
    _GENERAL_RESPONSE_TYPES = frozenset({'general', 'general_chat', 'current_info', 'planner_failure'})

    def __init__(
        self,
        enterprise_inference_service: ERPInferenceService,
        general_inference_service: GeneralInferenceService,
    ) -> None:
        self._enterprise_inference_service = enterprise_inference_service
        self._general_inference_service = general_inference_service

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._service_for(request).generate(request)

    def _service_for(self, request: GenerateRequest):
        response_type = (request.response_type or '').strip().casefold()
        if response_type in self._GENERAL_RESPONSE_TYPES:
            return self._general_inference_service
        return self._enterprise_inference_service
