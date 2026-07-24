"""Structured logging configuration."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.config.settings import Settings


LOG_FORMAT = (
    '%(asctime)s | %(levelname)-8s | %(logger_name)s | '
    'req=%(request_id)s | corr=%(correlation_id)s | conv=%(conversation_id)s | %(message)s'
)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.logger_name = record.name.replace('app.', '')
        try:
            from app.middleware.request_context import correlation_id_var, conversation_id_var, request_id_var

            record.request_id = request_id_var.get() or '-'
            record.correlation_id = correlation_id_var.get() or '-'
            record.conversation_id = conversation_id_var.get() or '-'
        except Exception:
            record.request_id = '-'
            record.correlation_id = '-'
            record.conversation_id = '-'
        return True


def configure_logging(settings: Settings) -> None:
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {'request_context': {'()': RequestContextFilter}},
            'formatters': {
                'default': {
                    'format': LOG_FORMAT,
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
