"""Business-layer chat service for standardized gateway responses."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.core.config import Settings
from app.core.exceptions import BadRequestError
from app.core.model_registry import ModelRegistry
from app.models.request import ChatRequest
from app.models.response import ChatCompletionResponse, ModelCard, ModelListResponse
from app.services.litellm_service import BaseInferenceProvider


class ChatService:
    """Business service responsible for validating and dispatching chat requests."""

    def __init__(
        self,
        inference_provider: BaseInferenceProvider,
        settings: Settings,
        model_registry: ModelRegistry,
    ) -> None:
        self._inference_provider = inference_provider
        self._settings = settings
        self._model_registry = model_registry

    async def generate_chat(self, request: ChatRequest) -> ChatCompletionResponse:
        """Resolve the effective model and forward the request to the provider."""

        model, profile = self._model_registry.resolve(request.model or self._settings.default_model)
        if request.stream:
            raise BadRequestError('Use the streaming endpoint flow for stream=true requests.')
        normalized_request = request.model_copy(update={'model': model})
        return await self._inference_provider.generate_chat(normalized_request, model=model, profile=profile)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Return an SSE event iterator for streaming chat completions."""

        model, profile = self._model_registry.resolve(request.model or self._settings.default_model)
        if not profile.supports_streaming:
            raise BadRequestError(f'Model "{model}" does not support streaming.')
        normalized_request = request.model_copy(update={'model': model, 'stream': True})
        return self._inference_provider.stream_chat(normalized_request, model=model, profile=profile)

    async def list_models(self) -> ModelListResponse:
        """Expose registered models using an OpenAI-compatible payload."""

        return ModelListResponse(
            data=[
                ModelCard(id=name, owned_by=f'{profile.provider}:{profile.backend}')
                for name, profile in self._model_registry.list_models().items()
            ]
        )
