import contextvars
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var = contextvars.ContextVar('request_id', default='')
correlation_id_var = contextvars.ContextVar('correlation_id', default='')
conversation_id_var = contextvars.ContextVar('conversation_id', default='')

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get('X-Request-Id', str(uuid.uuid4()))
        correlation_id = request.headers.get('X-Correlation-Id', request_id)
        conversation_id = request.headers.get('X-Conversation-Id', str(uuid.uuid4()))

        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.conversation_id = conversation_id

        request_id_var.set(request_id)
        correlation_id_var.set(correlation_id)
        conversation_id_var.set(conversation_id)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers['X-Request-Id'] = request_id
        response.headers['X-Correlation-Id'] = correlation_id
        response.headers['X-Conversation-Id'] = conversation_id
        response.headers['X-Response-Time-Ms'] = str(elapsed_ms)

        logger.info(
            'request_completed',
            extra={
                'path': request.url.path,
                'method': request.method,
                'status_code': response.status_code,
                'request_id': request_id,
                'correlation_id': correlation_id,
                'conversation_id': conversation_id,
                'execution_time_ms': elapsed_ms,
            },
        )
        return response
