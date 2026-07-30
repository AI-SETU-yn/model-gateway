"""Application-specific exception hierarchy."""

from __future__ import annotations


class AppException(Exception):
    """Base application exception with HTTP metadata."""

    status_code = 500
    code = 'INTERNAL_ERROR'

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class InvalidRequestError(AppException):
    status_code = 400
    code = 'INVALID_REQUEST'


class AdapterNotFoundError(AppException):
    status_code = 404
    code = 'ADAPTER_NOT_FOUND'


class InvalidAdapterError(AppException):
    status_code = 400
    code = 'INVALID_ADAPTER'


class ModelLoadError(AppException):
    status_code = 503
    code = 'MODEL_LOAD_ERROR'


class GPUUnavailableError(AppException):
    status_code = 503
    code = 'GPU_UNAVAILABLE'


class PlannerResponseError(AppException):
    status_code = 502
    code = 'PLANNER_RESPONSE_ERROR'


class InvalidProviderConfiguration(AppException):
    status_code = 500
    code = 'INVALID_PROVIDER_CONFIGURATION'


class UnsupportedProvider(InvalidProviderConfiguration):
    code = 'UNSUPPORTED_PROVIDER'


class MissingProviderApiKey(InvalidProviderConfiguration):
    code = 'MISSING_PROVIDER_API_KEY'


class MissingProviderEndpoint(InvalidProviderConfiguration):
    code = 'MISSING_PROVIDER_ENDPOINT'
