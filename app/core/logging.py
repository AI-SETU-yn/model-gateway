"""Structured logging setup for the Yn AI Setu Model Gateway."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.core.config import Settings
from app.middleware.request_id import request_id_var, span_id_var, trace_id_var

_LOG_FORMAT = (
    '%(asctime)s | %(levelname)-8s | %(name)s | '
    'request_id=%(request_id)s | trace_id=%(trace_id)s | span_id=%(span_id)s | %(message)s'
)


class RequestContextFilter(logging.Filter):
    """Inject request and trace context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or '-'
        record.trace_id = trace_id_var.get() or '-'
        record.span_id = span_id_var.get() or '-'
        return True


def configure_logging(settings: Settings) -> None:
    """Configure application-wide structured console logging."""

    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {'request_context': {'()': RequestContextFilter}},
            'formatters': {
                'default': {
                    'format': _LOG_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'filters': ['request_context'],
                }
            },
            'root': {'level': settings.log_level.upper(), 'handlers': ['console']},
        }
    )
