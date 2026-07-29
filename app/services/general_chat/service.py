from __future__ import annotations

from app.config.settings import ModelProfileConfig
from app.registry.prompt_registry import PromptRegistry
from app.schemas.inference import GenerateRequest, GenerateResponse, GeneralChatRequest
from app.services.inference import GeneralInferenceService
from app.services.prompt_builder import PromptBuilder


class GeneralChatService:
    def __init__(
        self,
        profile_name: str,
        profile: ModelProfileConfig,
        prompt_registry: PromptRegistry,
        inference_service: GeneralInferenceService,
    ) -> None:
        self._profile_name = profile_name
        self._profile = profile
        self._inference_service = inference_service
        self._prompt_builder = PromptBuilder(prompt_registry.general_chat_prompt(profile))

    async def chat(self, request: GeneralChatRequest) -> GenerateResponse:
        generate_request = GenerateRequest(
            adapter='',
            prompt=request.prompt,
            messages=request.messages,
            responseType=request.response_type or 'general_chat',
        )
        self._inference_service._prompt_builder = self._prompt_builder
        return await self._inference_service.generate(generate_request)
