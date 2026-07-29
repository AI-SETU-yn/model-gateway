from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, TypeVar

T = TypeVar('T')


class CircuitState(str, Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0


class CircuitBreakerOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._transition_if_recovery_elapsed()
            return self._state

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        with self._lock:
            self._transition_if_recovery_elapsed()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError('Circuit breaker is open.')
            if self._state == CircuitState.HALF_OPEN:
                pass
        try:
            result = await operation()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_at = time.monotonic()
            if self._failure_count >= self._config.failure_threshold:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._last_failure_at = 0.0

    def _transition_if_recovery_elapsed(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if (time.monotonic() - self._last_failure_at) >= self._config.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
