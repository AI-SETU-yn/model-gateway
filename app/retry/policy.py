from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

T = TypeVar('T')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    timeout_seconds: float = 90.0
    base_delay_seconds: float = 0.2
    backoff_multiplier: float = 2.0


class RetryExecutor:
    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy

    async def run(self, operation: Callable[[], Awaitable[T]], *, should_retry: Callable[[Exception], bool]) -> T:
        last_error: Exception | None = None
        for attempt in range(self._policy.max_retries + 1):
            try:
                return await asyncio.wait_for(operation(), timeout=self._policy.timeout_seconds)
            except Exception as exc:
                last_error = exc
                retry_allowed = attempt < self._policy.max_retries and should_retry(exc)
                logger.warning(
                    'retry_executor_failure attempt=%s max_attempts=%s retry=%s error_type=%s error=%s',
                    attempt + 1,
                    self._policy.max_retries + 1,
                    retry_allowed,
                    type(exc).__name__,
                    str(exc),
                )
                if not retry_allowed:
                    raise
                delay = self._policy.base_delay_seconds * (self._policy.backoff_multiplier ** attempt)
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error
