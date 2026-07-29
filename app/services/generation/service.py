"""Response generation service for ERP model flows."""

from __future__ import annotations

from app.schemas.inference import GenerateRequest, GenerateResponse
from app.services.inference import ERPInferenceService


class ResponseGenerationService:
    def __init__(self, inference_service: ERPInferenceService) -> None:
        self._inference_service = inference_service

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._inference_service.generate(request)
