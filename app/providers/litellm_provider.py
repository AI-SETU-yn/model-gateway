from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from app.clients.base import BaseInferenceClient
from app.models.inference import InferenceRequest, InferenceResponse, InferenceUsage
from app.observability.inference import InferenceObserver, elapsed_ms_since
from app.providers.base import BaseInferenceProvider
from app.retry.circuit_breaker import CircuitBreaker
from app.retry.policy import RetryExecutor

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseInferenceProvider):
    def __init__(
        self,
        client: BaseInferenceClient,
        retry_executor: RetryExecutor,
        observer: InferenceObserver,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._client = client
        self._retry_executor = retry_executor
        self._observer = observer
        self._circuit_breaker = circuit_breaker

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()

        async def operation() -> InferenceResponse:
            return await self._retry_executor.run(lambda: self._client.completion(request), should_retry=self._should_retry)

        try:
            response = await self._circuit_breaker.call(operation)
            response = response.model_copy(update={'latencyMs': elapsed_ms_since(started)})
            self._observer.on_success(request, response)
            return response
        except Exception as exc:
            logger.warning(
                'litellm_provider_failure request_id=%s provider=%s base_url=%s model_alias=%s model_name=%s adapter=%s circuit_state=%s error_type=%s error=%s',
                str(request.metadata.get('request_id') or '-') or '-',
                str(request.metadata.get('provider') or '-') or '-',
                str(request.metadata.get('api_base') or '-') or '-',
                request.model_alias,
                str(request.metadata.get('model_name') or request.model_alias),
                request.adapter or '-',
                self._circuit_breaker.state.value,
                type(exc).__name__,
                str(exc),
            )
            self._observer.on_failure(
                request,
                provider=str(request.metadata.get('provider') or ''),
                model=str(request.metadata.get('model_name') or request.model_alias),
                latency_ms=elapsed_ms_since(started),
                error=exc,
            )
            raise

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        started = time.perf_counter()
        stream_request = request.model_copy(update={'stream': True})
        aggregated_text: list[str] = []
        final_chunk: InferenceResponse | None = None

        async def operation() -> AsyncIterator[InferenceResponse]:
            return await self._retry_executor.run(lambda: self._client.stream_completion(stream_request), should_retry=self._should_retry)

        try:
            response_stream = await self._circuit_breaker.call(operation)
            async for chunk in response_stream:
                final_chunk = chunk
                if chunk.content:
                    aggregated_text.append(chunk.content)
                yield chunk.model_copy(update={'latencyMs': elapsed_ms_since(started)})
            observer_response = (final_chunk or InferenceResponse(
                content=''.join(aggregated_text),
                finishReason='stop',
                usage=InferenceUsage(promptTokens=0, completionTokens=0, totalTokens=0),
                latencyMs=elapsed_ms_since(started),
                provider=str(request.metadata.get('provider') or ''),
                model=str(request.metadata.get('model_name') or request.model_alias),
                adapter=request.adapter,
                requestId=str(request.metadata.get('request_id') or ''),
                traceId=str(request.metadata.get('trace_id') or ''),
                stream=True,
            )).model_copy(update={'content': ''.join(aggregated_text), 'latencyMs': elapsed_ms_since(started)})
            self._observer.on_success(stream_request, observer_response)
        except Exception as exc:
            logger.warning(
                'litellm_provider_stream_failure request_id=%s provider=%s base_url=%s model_alias=%s model_name=%s adapter=%s circuit_state=%s error_type=%s error=%s',
                str(request.metadata.get('request_id') or '-') or '-',
                str(request.metadata.get('provider') or '-') or '-',
                str(request.metadata.get('api_base') or '-') or '-',
                request.model_alias,
                str(request.metadata.get('model_name') or request.model_alias),
                request.adapter or '-',
                self._circuit_breaker.state.value,
                type(exc).__name__,
                str(exc),
            )
            self._observer.on_failure(
                stream_request,
                provider=str(request.metadata.get('provider') or ''),
                model=str(request.metadata.get('model_name') or request.model_alias),
                latency_ms=elapsed_ms_since(started),
                error=exc,
            )
            raise

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        transient_names = {'TimeoutError', 'ConnectError', 'ReadTimeout', 'APITimeoutError', 'CircuitBreakerOpenError'}
        return type(exc).__name__ in transient_names
