from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from app.clients.base import BaseInferenceClient
from app.models.inference import InferenceRequest, InferenceResponse
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
        yield await self.complete(request.model_copy(update={'stream': True}))

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        transient_names = {'TimeoutError', 'ConnectError', 'ReadTimeout', 'APITimeoutError', 'CircuitBreakerOpenError'}
        return type(exc).__name__ in transient_names
