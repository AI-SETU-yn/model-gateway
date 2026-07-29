from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.models.inference import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceObservation:
    request_id: str | None
    trace_id: str | None
    model: str
    provider: str
    adapter: str | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    success: bool


class InferenceObserver:
    def on_success(self, request: InferenceRequest, response: InferenceResponse) -> None:
        observation = InferenceObservation(
            request_id=response.request_id,
            trace_id=response.trace_id,
            model=response.model,
            provider=response.provider,
            adapter=response.adapter,
            latency_ms=response.latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=response.finish_reason,
            success=True,
        )
        logger.info('inference_success', extra=observation.__dict__)

    def on_failure(self, request: InferenceRequest, *, provider: str, model: str, latency_ms: float, error: Exception) -> None:
        observation = InferenceObservation(
            request_id=str(request.metadata.get('request_id') or ''),
            trace_id=str(request.metadata.get('trace_id') or ''),
            model=model,
            provider=provider,
            adapter=request.adapter,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason=type(error).__name__,
            success=False,
        )
        logger.warning('inference_failure', extra=observation.__dict__)


def elapsed_ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
