"""Application exception hierarchy and FastAPI exception handlers."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.response import ErrorResponse

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """Base exception for all gateway failures."""

    status_code = 500
    code = 'INTERNAL_ERROR'

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class BadRequestError(GatewayError):
    status_code = 400
    code = 'BAD_REQUEST'


class UnauthorizedError(GatewayError):
    status_code = 401
    code = 'UNAUTHORIZED'


class ResourceNotFoundError(GatewayError):
    status_code = 404
    code = 'NOT_FOUND'


class RequestTimeoutError(GatewayError):
    status_code = 408
    code = 'REQUEST_TIMEOUT'


class RateLimitError(GatewayError):
    status_code = 429
    code = 'RATE_LIMITED'


class UpstreamServerError(GatewayError):
    status_code = 502
    code = 'UPSTREAM_ERROR'


class ServiceUnavailableError(GatewayError):
    status_code = 503
    code = 'SERVICE_UNAVAILABLE'


# `AppException` is the name used throughout the per-capability inference
# subsystem (registry/model_registry.py, services/planner.py, ...); it is an
# alias rather than a parallel hierarchy so both names share one set of
# FastAPI exception handlers.
AppException = GatewayError


class MissingProviderApiKey(GatewayError):
    status_code = 500
    code = 'MISSING_PROVIDER_API_KEY'


class MissingProviderEndpoint(GatewayError):
    status_code = 500
    code = 'MISSING_PROVIDER_ENDPOINT'


class PlannerResponseError(UpstreamServerError):
    code = 'PLANNER_RESPONSE_ERROR'


class InvalidProviderConfiguration(UpstreamServerError):
    code = 'INVALID_PROVIDER_CONFIGURATION'


def _build_error_response(request: Request, exc: GatewayError) -> ErrorResponse:
    return ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=getattr(request.state, 'request_id', None),
        trace_id=getattr(request.state, 'trace_id', None),
        span_id=getattr(request.state, 'span_id', None),
    )


async def gateway_exception_handler(request: Request, exc: GatewayError) -> JSONResponse:
    logger.warning('gateway_exception path=%s code=%s message=%s', request.url.path, exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=_build_error_response(request, exc).model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning('request_validation_error path=%s errors=%s', request.url.path, exc.errors())
    body = ErrorResponse(
        code='BAD_REQUEST',
        message='Invalid request payload.',
        request_id=getattr(request.state, 'request_id', None),
        trace_id=getattr(request.state, 'trace_id', None),
        span_id=getattr(request.state, 'span_id', None),
    )
    return JSONResponse(status_code=400, content=body.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception('unhandled_exception path=%s', request.url.path)
    body = ErrorResponse(
        code='INTERNAL_ERROR',
        message='An unexpected error occurred.',
        request_id=getattr(request.state, 'request_id', None),
        trace_id=getattr(request.state, 'trace_id', None),
        span_id=getattr(request.state, 'span_id', None),
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""

    app.add_exception_handler(GatewayError, gateway_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
