from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from app.config.settings import ModelGatewayConfig
from app.models.inference import InferenceRequest, InferenceResponse
from app.providers.base import BaseInferenceProvider


class ConfiguredInferenceProvider(BaseInferenceProvider):
    def __init__(
        self,
        model_config: ModelGatewayConfig,
        transformers_provider: Callable[[], BaseInferenceProvider],
        litellm_provider: Callable[[], BaseInferenceProvider],
    ) -> None:
        self._model_config = model_config
        self._transformers_provider = transformers_provider
        self._litellm_provider = litellm_provider

    def _provider_for(self, request: InferenceRequest) -> BaseInferenceProvider:
        profile = self._model_config.get_profile(request.model_alias)
        provider_name = profile.provider.strip().lower()
        if provider_name == 'transformers':
            return self._transformers_provider()
        return self._litellm_provider()

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        return await self._provider_for(request).complete(request)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        async for chunk in self._provider_for(request).stream(request):
            yield chunk
