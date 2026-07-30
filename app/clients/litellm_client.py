from __future__ import annotations

import logging
from typing import Any

import litellm

from app.clients.base import BaseInferenceClient
from app.config.settings import ModelGatewayConfig, Settings
from app.exceptions.errors import InvalidProviderConfiguration
from app.models.inference import InferenceRequest, InferenceResponse, InferenceUsage

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES = {
    'together_ai': 'togetherai',
    'huggingface_endpoint': 'huggingface',
    'vllm': 'openai',
}


class LiteLLMClient(BaseInferenceClient):
    def __init__(self, settings: Settings, model_config: ModelGatewayConfig) -> None:
        self._settings = settings
        self._model_config = model_config
        litellm.drop_params = True

    async def completion(self, request: InferenceRequest) -> InferenceResponse:
        profile = self._model_config.get_profile(request.model_alias)
        provider_type = (profile.provider or str(request.metadata.get('provider') or '')).strip().lower()
        api_base = (profile.base_url or str(request.metadata.get('api_base') or '')).strip()
        api_key = (profile.api_key or str(request.metadata.get('api_key') or '')).strip()
        model_name = (profile.deployment_name or profile.model_name or str(request.metadata.get('model_name') or request.model_alias)).strip()
        request_id = str(request.metadata.get('request_id') or '')
        kwargs: dict[str, Any] = {
            'model': model_name,
            'messages': request.to_messages(),
            'timeout': self._settings.litellm_timeout_seconds,
            'metadata': request.metadata,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'stream': request.stream,
        }
        provider_name = _PROVIDER_ALIASES.get(provider_type, provider_type)
        if provider_name:
            kwargs['custom_llm_provider'] = provider_name
        if api_key:
            kwargs['api_key'] = api_key
        if api_base:
            kwargs['api_base'] = api_base
        if profile.api_version:
            kwargs['api_version'] = profile.api_version
        if profile.organization:
            kwargs['organization'] = profile.organization
        if request.adapter:
            kwargs['extra_body'] = {'adapter': request.adapter}
        logger.info(
            'litellm_request request_id=%s provider=%s base_url=%s model_alias=%s model_name=%s adapter=%s endpoint=%s',
            request_id or '-',
            provider_name or '-',
            api_base or '-',
            request.model_alias,
            model_name,
            request.adapter or '-',
            'chat.completions',
        )
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise InvalidProviderConfiguration(f'LiteLLM request failed for provider "{provider_type}" and model alias "{request.model_alias}": {exc}') from exc
        return self._normalize_response(response, request, provider_name, model_name)

    @staticmethod
    def _normalize_response(response: Any, request: InferenceRequest, provider_name: str, model_name: str) -> InferenceResponse:
        choice = response.choices[0]
        message = getattr(choice, 'message', choice.get('message'))
        content = getattr(message, 'content', None) if not isinstance(message, dict) else message.get('content')
        usage = getattr(response, 'usage', None)
        finish_reason = getattr(choice, 'finish_reason', None) if not isinstance(choice, dict) else choice.get('finish_reason')
        return InferenceResponse(
            content=str(content or ''),
            finishReason=finish_reason,
            usage=InferenceUsage(
                promptTokens=int(getattr(usage, 'prompt_tokens', 0) if usage is not None else 0),
                completionTokens=int(getattr(usage, 'completion_tokens', 0) if usage is not None else 0),
                totalTokens=int(getattr(usage, 'total_tokens', 0) if usage is not None else 0),
            ),
            latencyMs=0.0,
            provider=provider_name,
            model=str(getattr(response, 'model', '') or model_name or request.model_alias),
            adapter=request.adapter,
            requestId=str(request.metadata.get('request_id') or ''),
            traceId=str(request.metadata.get('trace_id') or ''),
            stream=request.stream,
        )
