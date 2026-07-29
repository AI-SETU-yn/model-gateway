from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

T = TypeVar('T')


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
                if attempt >= self._policy.max_retries or not should_retry(exc):
                    raise
                delay = self._policy.base_delay_seconds * (self._policy.backoff_multiplier ** attempt)
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error
