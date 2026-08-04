"""OpenTelemetry helpers for request and inference tracing."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback when package is absent
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False

_TELEMETRY_CONFIGURED = False


class _NoOpSpanContext:
    trace_id = 0
    span_id = 0


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: Exception) -> None:
        return None

    def get_span_context(self) -> _NoOpSpanContext:
        return _NoOpSpanContext()


class _NoOpSpanManager:
    def __enter__(self) -> _NoOpSpan:
        return _NoOpSpan()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _NoOpTracer:
    def start_as_current_span(self, name: str) -> _NoOpSpanManager:
        return _NoOpSpanManager()


def configure_telemetry(service_name: str) -> bool:
    """Configure a minimal OpenTelemetry tracer provider."""

    global _TELEMETRY_CONFIGURED
    if not _OTEL_AVAILABLE:
        logger.warning('opentelemetry_not_installed_tracing_disabled')
        return False
    if _TELEMETRY_CONFIGURED:
        return True

    provider = TracerProvider(resource=Resource.create({'service.name': service_name}))
    trace.set_tracer_provider(provider)
    _TELEMETRY_CONFIGURED = True
    return True


def get_tracer(name: str):
    """Return a configured tracer or a no-op tracer."""

    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    return trace.get_tracer(name)


def get_trace_context_ids(span: Any) -> tuple[str, str]:
    """Extract trace and span IDs as zero-padded hex strings."""

    context = span.get_span_context()
    trace_id = format(getattr(context, 'trace_id', 0), '032x')
    span_id = format(getattr(context, 'span_id', 0), '016x')
    return trace_id, span_id
