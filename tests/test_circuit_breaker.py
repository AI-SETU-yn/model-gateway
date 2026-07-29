import asyncio
import time

import pytest

from app.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError, CircuitState


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=0.01))

    async def fail():
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(fail)


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_to_closed() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01))

    async def fail():
        raise RuntimeError('boom')

    async def succeed():
        return 'ok'

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    time.sleep(0.02)
    result = await breaker.call(succeed)
    assert result == 'ok'
    assert breaker.state == CircuitState.CLOSED
