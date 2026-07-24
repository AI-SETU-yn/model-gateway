from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions.errors import AppException
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.warning('app_exception', extra={'code': exc.code, 'message': exc.message, 'path': request.url.path})
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning('request_validation_error', extra={'path': request.url.path, 'errors': exc.errors()})
    body = ErrorResponse(
        code='VALIDATION_ERROR',
        message='Invalid request payload.',
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception('unhandled_exception', extra={'path': request.url.path})
    body = ErrorResponse(
        code='INTERNAL_ERROR',
        message='An unexpected error occurred.',
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=500, content=body.model_dump())
