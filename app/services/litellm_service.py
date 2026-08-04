"""LiteLLM-backed provider service for chat completions."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    BadRequestError,
    GatewayError,
    RateLimitError,
    RequestTimeoutError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
    UpstreamServerError,
)
from app.core.model_registry import ModelRegistry, RegisteredModel
from app.middleware.request_id import request_id_var
from app.models.request import ChatRequest
from app.models.response import ChatCompletionResponse
from app.utils.telemetry import get_trace_context_ids, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
_litellm_module: Any | None = None


@dataclass(frozen=True)
class ReadinessResult:
    """Aggregated provider readiness state."""

    healthy: bool
    checks: dict[str, Any]


class BaseInferenceProvider(ABC):
    """Abstraction for future provider implementations."""

    @abstractmethod
    async def generate_chat(
        self,
        request: ChatRequest,
        *,
        model: str,
        profile: RegisteredModel,
    ) -> ChatCompletionResponse:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self,
        request: ChatRequest,
        *,
        model: str,
        profile: RegisteredModel,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def check_connectivity(self) -> ReadinessResult:
        raise NotImplementedError


class LiteLLMService(BaseInferenceProvider):
    """Provider implementation that talks to LiteLLM instead of vLLM directly."""

    def __init__(self, settings: Settings, model_registry: ModelRegistry, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._model_registry = model_registry
        self._http_client = http_client

    async def generate_chat(
        self,
        request: ChatRequest,
        *,
        model: str,
        profile: RegisteredModel,
    ) -> ChatCompletionResponse:
        """Send a chat completion request to LiteLLM with retries and tracing."""

        await self._validate_registered_model(model, profile)
        last_error: GatewayError | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            with tracer.start_as_current_span('litellm.generate_chat') as span:
                trace_id, span_id = get_trace_context_ids(span)
                span.set_attribute('llm.model', model)
                span.set_attribute('llm.provider', profile.provider)
                span.set_attribute('llm.backend', profile.backend)
                span.set_attribute('llm.endpoint', profile.endpoint)
                span.set_attribute('llm.request_id', request_id_var.get() or '-')
                span.set_attribute('llm.retry_attempt', attempt)

                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        self._acompletion(request, model=model, profile=profile),
                        timeout=self._settings.request_timeout,
                    )
                    litellm_latency_ms = round((time.perf_counter() - started) * 1000, 2)
                    normalized = self._normalize_response(response)
                    span.set_attribute('llm.trace_id', trace_id)
                    span.set_attribute('llm.span_id', span_id)
                    span.set_attribute('llm.latency_ms', litellm_latency_ms)
                    span.set_attribute('llm.prompt_tokens', normalized.usage.prompt_tokens)
                    span.set_attribute('llm.completion_tokens', normalized.usage.completion_tokens)
                    span.set_attribute('llm.total_tokens', normalized.usage.total_tokens)
                    logger.info(
                        'litellm_chat_completed request_id=%s model=%s provider=%s backend=%s latency_ms=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
                        request_id_var.get() or '-',
                        normalized.model,
                        profile.provider,
                        profile.backend,
                        litellm_latency_ms,
                        normalized.usage.prompt_tokens,
                        normalized.usage.completion_tokens,
                        normalized.usage.total_tokens,
                    )
                    return normalized
                except Exception as exc:  # pragma: no cover
                    mapped = self._map_exception(exc)
                    last_error = mapped
                    span.record_exception(exc)
                    span.set_attribute('error.type', type(mapped).__name__)
                    span.set_attribute('error.message', mapped.message)
                    should_retry = self._should_retry(mapped, attempt)
                    logger.warning(
                        'litellm_chat_failed request_id=%s model=%s attempt=%s retry=%s error_type=%s error=%s',
                        request_id_var.get() or '-',
                        model,
                        attempt,
                        should_retry,
                        type(mapped).__name__,
                        mapped.message,
                    )
                    if not should_retry:
                        raise mapped
                    await asyncio.sleep(self._backoff_seconds(attempt))
        raise last_error or ServiceUnavailableError('LiteLLM request failed after retries.')

    def stream_chat(
        self,
        request: ChatRequest,
        *,
        model: str,
        profile: RegisteredModel,
    ) -> AsyncIterator[str]:
        """Return a server-sent event stream backed by LiteLLM streaming completions."""

        async def event_generator() -> AsyncIterator[str]:
            await self._validate_registered_model(model, profile)
            started = time.perf_counter()
            try:
                stream = await self._acompletion(request, model=model, profile=profile, stream=True)
                async for chunk in stream:
                    payload = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
                    yield f'data: {json.dumps(payload)}\n\n'
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.info(
                    'litellm_chat_stream_completed request_id=%s model=%s provider=%s latency_ms=%s',
                    request_id_var.get() or '-',
                    model,
                    profile.provider,
                    latency_ms,
                )
                yield 'data: [DONE]\n\n'
            except Exception as exc:  # pragma: no cover
                mapped = self._map_exception(exc)
                logger.warning(
                    'litellm_chat_stream_failed request_id=%s model=%s error_type=%s error=%s',
                    request_id_var.get() or '-',
                    model,
                    type(mapped).__name__,
                    mapped.message,
                )
                raise mapped

        return event_generator()

    async def check_connectivity(self) -> ReadinessResult:
        """Verify LiteLLM reachability, remote model listing, and inference readiness."""

        checks: dict[str, Any] = {
            'configured': True,
            'registry_loaded': True,
            'registry_path': str(self._model_registry.path.resolve()),
            'litellm_reachable': False,
            'models_endpoint_reachable': False,
            'default_model_registered': False,
            'default_model_remote_available': False,
            'remote_inference_ready': False,
            'base_url': self._settings.litellm_base_url,
            'default_model': self._settings.default_model,
        }

        remote_models: set[str] = set()
        try:
            response = await self._http_client.get(self._settings.models_url, headers=self._headers())
            if response.status_code < 500:
                checks['litellm_reachable'] = True
            response.raise_for_status()
            checks['models_endpoint_reachable'] = True
            payload = response.json()
            remote_models = {item.get('id', '') for item in payload.get('data', []) if item.get('id')}
            checks['remote_models'] = sorted(remote_models)
        except Exception as exc:
            mapped = self._map_exception(exc)
            checks['models_error'] = mapped.message

        try:
            _, profile = self._model_registry.resolve(self._settings.default_model)
            checks['default_model_registered'] = True
            if self._settings.default_model in remote_models:
                checks['default_model_remote_available'] = True
            await self._validate_registered_model(self._settings.default_model, profile, remote_models=remote_models)
            probe_request = ChatRequest(
                messages=[{'role': 'user', 'content': 'ping'}],
                model=self._settings.default_model,
                temperature=0.0,
                max_tokens=1,
            )
            await self.generate_chat(probe_request, model=self._settings.default_model, profile=profile)
            checks['remote_inference_ready'] = True
        except GatewayError as exc:
            checks['inference_error'] = exc.message

        healthy = all(
            bool(checks[key])
            for key in (
                'configured',
                'registry_loaded',
                'litellm_reachable',
                'models_endpoint_reachable',
                'default_model_registered',
                'default_model_remote_available',
                'remote_inference_ready',
            )
        )
        return ReadinessResult(healthy=healthy, checks=checks)

    async def _acompletion(
        self,
        request: ChatRequest,
        *,
        model: str,
        profile: RegisteredModel,
        stream: bool = False,
    ) -> Any:
        litellm = self._get_litellm()
        kwargs = {
            'api_base': profile.endpoint,
            'api_key': self._settings.api_key,
            'model': model,
            'messages': [message.model_dump() for message in request.messages],
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'stream': stream,
        }
        return await litellm.acompletion(**kwargs)

    async def _validate_registered_model(
        self,
        model: str,
        profile: RegisteredModel,
        *,
        remote_models: set[str] | None = None,
    ) -> None:
        if profile.provider.lower() != 'litellm':
            raise ServiceUnavailableError(f'Model "{model}" is configured with unsupported provider "{profile.provider}".')
        if remote_models is None:
            remote_models = await self._fetch_remote_model_ids()
        if model not in remote_models:
            raise ResourceNotFoundError(
                f'Model "{model}" is registered in {self._model_registry.path.name} but not exposed by LiteLLM/vLLM.'
            )

    async def _fetch_remote_model_ids(self) -> set[str]:
        response = await self._http_client.get(self._settings.models_url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        return {item.get('id', '') for item in payload.get('data', []) if item.get('id')}

    def _headers(self) -> dict[str, str] | None:
        return {'Authorization': f'Bearer {self._settings.api_key}'} if self._settings.api_key else None

    @staticmethod
    def _normalize_response(raw_response: Any) -> ChatCompletionResponse:
        payload = raw_response.model_dump() if hasattr(raw_response, 'model_dump') else raw_response
        return ChatCompletionResponse.model_validate(payload)

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        return min(0.5 * (2 ** (attempt - 1)), 8.0)

    def _should_retry(self, exc: GatewayError, attempt: int) -> bool:
        return attempt < self._settings.max_retries and exc.status_code in {408, 429, 502, 503}

    @staticmethod
    def _map_exception(exc: Exception) -> GatewayError:
        if isinstance(exc, GatewayError):
            return exc
        if isinstance(exc, asyncio.TimeoutError):
            return RequestTimeoutError('LiteLLM request timed out.')
        if isinstance(exc, httpx.TimeoutException):
            return RequestTimeoutError('LiteLLM request timed out.')
        if isinstance(exc, httpx.ConnectError):
            return ServiceUnavailableError('Unable to connect to LiteLLM.')
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            message = exc.response.text or str(exc)
        else:
            status_code = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
            message = str(exc)
        if status_code == 400:
            return BadRequestError(message or 'LiteLLM rejected the request.')
        if status_code == 401:
            return UnauthorizedError(message or 'LiteLLM authentication failed.')
        if status_code == 404:
            return ResourceNotFoundError(message or 'Requested model was not found.')
        if status_code == 408:
            return RequestTimeoutError(message or 'LiteLLM request timed out.')
        if status_code == 429:
            return RateLimitError(message or 'LiteLLM rate limit exceeded.')
        if status_code == 503:
            return ServiceUnavailableError(message or 'LiteLLM service unavailable.')
        if status_code in {500, 502, 504}:
            return UpstreamServerError(message or 'LiteLLM upstream failure.')
        return UpstreamServerError(message or 'Unexpected LiteLLM failure.')

    @staticmethod
    def _get_litellm() -> Any:
        global _litellm_module
        if _litellm_module is None:
            _litellm_module = importlib.import_module('litellm')
            _litellm_module.drop_params = True
        return _litellm_module
