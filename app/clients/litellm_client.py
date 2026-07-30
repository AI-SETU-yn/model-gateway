from __future__ import annotations

import logging
from typing import Any

import litellm

from app.clients.base import BaseInferenceClient
from app.config.settings import Settings
from app.models.inference import InferenceRequest, InferenceResponse, InferenceUsage

logger = logging.getLogger(__name__)


class LiteLLMClient(BaseInferenceClient):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        litellm.drop_params = True

    async def completion(self, request: InferenceRequest) -> InferenceResponse:
        request_id = str(request.metadata.get('request_id') or '')
        provider = str(request.metadata.get('provider') or '')
        api_base = str(request.metadata.get('api_base') or '')
        model_name = str(request.metadata.get('model_name') or request.model_alias)
        logger.info(
            'litellm_request request_id=%s provider=%s base_url=%s model_alias=%s model_name=%s adapter=%s endpoint=%s',
            request_id or '-',
            provider or '-',
            api_base or '-',
            request.model_alias,
            model_name,
            request.adapter or '-',
            'chat.completions',
        )
        response = await litellm.acompletion(
            model=model_name,
            custom_llm_provider=provider,
            api_base=api_base,
            api_key=str(request.metadata.get('api_key') or ''),
            messages=request.to_messages(),
            timeout=self._settings.litellm_timeout_seconds,
            metadata=request.metadata,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            extra_body={'adapter': request.adapter} if request.adapter else None,
        )
        return self._normalize_response(response, request)

    @staticmethod
    def _normalize_response(response: Any, request: InferenceRequest) -> InferenceResponse:
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
            provider=str(request.metadata.get('provider') or ''),
            model=str(getattr(response, 'model', '') or request.metadata.get('model_name') or request.model_alias),
            adapter=request.adapter,
            requestId=str(request.metadata.get('request_id') or ''),
            traceId=str(request.metadata.get('trace_id') or ''),
            stream=request.stream,
        )
