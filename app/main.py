from contextlib import asynccontextmanager
from pathlib import Path
import os
import time

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.routers.inference import router as inference_router
from app.api.routers.management import router as management_router
from app.config.settings import get_model_config, get_settings
from app.exceptions.errors import AppException
from app.exceptions.handlers import app_exception_handler, request_validation_exception_handler, unhandled_exception_handler
from app.middleware.request_context import RequestContextMiddleware
from app.services.dependencies import get_health_service, get_model_registry
from app.utils.logging import configure_logging

ENV_VERIFICATION_KEYS = ('BASE_MODEL', 'MODEL_NAME', 'MODEL_ID', 'HF_MODEL', 'BASE_MODEL_NAME', 'HUGGINGFACE_MODEL')


def _config_path(settings) -> Path:
    return (Path.cwd() / settings.config_path).resolve()


def _log_environment_verification() -> None:
    for key in ENV_VERIFICATION_KEYS:
        value = os.environ.get(key)
        print(f'Environment verification: {key}={value if value is not None else "Not Set"}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = get_model_config()
    configure_logging(settings)
    app.state.is_ready = False
    app.state.startup_latency_ms = 0.0
    started = time.perf_counter()
    loaded_config_path = _config_path(settings)
    print(f'Loaded configuration: {loaded_config_path}')
    print(f'Current working directory: {Path.cwd().resolve()}')
    for profile_name, profile in config.models.items():
        print(f'Model profile: {profile_name} -> model={profile.model_name} provider={profile.provider} base_model={profile.base_model}')
    _log_environment_verification()
    registry = get_model_registry()
    try:
        await get_health_service().check_all(registry.all_profiles())
    except Exception as exc:
        print(f'Startup warm-up failed: {type(exc).__name__}: {exc}')
    app.state.startup_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    app.state.is_ready = settings.ready_on_startup
    yield
    app.state.is_ready = False


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(inference_router)
    app.include_router(management_router)
    return app


app = create_app()
