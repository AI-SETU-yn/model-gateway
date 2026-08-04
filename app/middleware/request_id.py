"""Request ID middleware with trace propagation support."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.utils.telemetry import get_trace_context_ids, get_tracer

request_id_var = contextvars.ContextVar('request_id', default='')
trace_id_var = contextvars.ContextVar('trace_id', default='')
span_id_var = contextvars.ContextVar('span_id', default='')

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and tracing context to each incoming request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request_token = request_id_var.set(request_id)
        start = time.perf_counter()

        with tracer.start_as_current_span(f'{request.method} {request.url.path}') as span:
            trace_id, span_id = get_trace_context_ids(span)
            trace_token = trace_id_var.set(trace_id)
            span_token = span_id_var.set(span_id)

            request.state.request_id = request_id
            request.state.trace_id = trace_id
            request.state.span_id = span_id

            span.set_attribute('http.method', request.method)
            span.set_attribute('http.route', request.url.path)
            span.set_attribute('request.id', request_id)

            try:
                response = await call_next(request)
            except Exception as exc:
                span.record_exception(exc)
                raise
            finally:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info('request_completed path=%s method=%s latency_ms=%s', request.url.path, request.method, elapsed_ms)
                request_id_var.reset(request_token)
                trace_id_var.reset(trace_token)
                span_id_var.reset(span_token)

            response.headers['X-Request-ID'] = request_id
            response.headers['X-Trace-ID'] = trace_id
            response.headers['X-Span-ID'] = span_id
            response.headers['X-Response-Time-Ms'] = str(elapsed_ms)
            return response
