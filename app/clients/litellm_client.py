from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator
from typing import Any

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

_litellm_module: Any | None = None


def _get_litellm() -> Any:
    global _litellm_module
    if _litellm_module is None:
        _litellm_module = importlib.import_module('litellm')
        _litellm_module.drop_params = True
    return _litellm_module


class LiteLLMClient(BaseInferenceClient):
    def __init__(self, settings: Settings, model_config: ModelGatewayConfig) -> None:
        self._settings = settings
        self._model_config = model_config

    async def completion(self, request: InferenceRequest) -> InferenceResponse:
        kwargs, provider_type, provider_name, model_name = self._build_request_kwargs(request)
        request_id = str(request.metadata.get('request_id') or '')
        self._log_request(request, provider_name, str(kwargs.get('api_base') or ''), model_name, request_id)
        try:
            response = await _get_litellm().acompletion(**kwargs)
        except Exception as exc:
            raise InvalidProviderConfiguration(f'LiteLLM request failed for provider "{provider_type}" and model alias "{request.model_alias}": {exc}') from exc
        return self._normalize_response(response, request, provider_name or provider_type, model_name)

    async def stream_completion(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        stream_request = request.model_copy(update={'stream': True})
        kwargs, provider_type, provider_name, model_name = self._build_request_kwargs(stream_request)
        request_id = str(stream_request.metadata.get('request_id') or '')
        self._log_request(stream_request, provider_name, str(kwargs.get('api_base') or ''), model_name, request_id)
        try:
            response_stream = await _get_litellm().acompletion(**kwargs)
        except Exception as exc:
            raise InvalidProviderConfiguration(f'LiteLLM stream request failed for provider "{provider_type}" and model alias "{stream_request.model_alias}": {exc}') from exc
        async for chunk in response_stream:
            normalized = self._normalize_stream_chunk(chunk, stream_request, provider_name or provider_type, model_name)
            if normalized is not None:
                yield normalized

    def _build_request_kwargs(self, request: InferenceRequest) -> tuple[dict[str, Any], str, str, str]:
        profile = self._model_config.get_profile(request.model_alias)
        provider_config = profile.provider_config
        provider_type = profile.provider_type
        api_base = (provider_config.api_base or str(request.metadata.get('api_base') or '')).strip()
        api_key = (provider_config.api_key or str(request.metadata.get('api_key') or '')).strip()
        model_name = (provider_config.deployment_name or profile.model_name or str(request.metadata.get('model_name') or request.model_alias)).strip()
        kwargs: dict[str, Any] = {
            'model': model_name,
            'messages': request.to_messages(),
            'timeout': provider_config.timeout_seconds or self._settings.litellm_timeout_seconds,
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
        if provider_config.api_version:
            kwargs['api_version'] = provider_config.api_version
        if provider_config.organization:
            kwargs['organization'] = provider_config.organization
        if request.adapter:
            kwargs['extra_body'] = {'adapter': request.adapter}
        return kwargs, provider_type, provider_name, model_name

    @staticmethod
    def _log_request(request: InferenceRequest, provider_name: str, api_base: str, model_name: str, request_id: str) -> None:
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

    @staticmethod
    def _normalize_response(response: Any, request: InferenceRequest, provider_name: str, model_name: str) -> InferenceResponse:
        choice = response.choices[0]
        message = getattr(choice, 'message', None) if not isinstance(choice, dict) else choice.get('message')
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

    @staticmethod
    def _normalize_stream_chunk(chunk: Any, request: InferenceRequest, provider_name: str, model_name: str) -> InferenceResponse | None:
        choice = chunk.choices[0] if not isinstance(chunk, dict) else chunk.get('choices', [{}])[0]
        delta = getattr(choice, 'delta', None) if not isinstance(choice, dict) else choice.get('delta')
        content = delta.get('content') if isinstance(delta, dict) else getattr(delta, 'content', None)
        finish_reason = getattr(choice, 'finish_reason', None) if not isinstance(choice, dict) else choice.get('finish_reason')
        usage = getattr(chunk, 'usage', None)
        if not content and finish_reason is None and usage is None:
            return None
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
            model=str(getattr(chunk, 'model', '') or model_name or request.model_alias),
            adapter=request.adapter,
            requestId=str(request.metadata.get('request_id') or ''),
            traceId=str(request.metadata.get('trace_id') or ''),
            stream=True,
        )
