"""FastAPI application bootstrap for the Yn AI Setu Model Gateway."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_http_client, get_litellm_service, get_settings
from app.core.exceptions import ServiceUnavailableError, register_exception_handlers
from app.core.logging import configure_logging
from app.middleware.request_id import RequestIDMiddleware
from app.utils.telemetry import configure_telemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize logging, telemetry, provider readiness, and graceful shutdown."""

    settings = get_settings()
    configure_logging(settings)
    configure_telemetry(settings.app_name)

    app.state.ready = False
    readiness = await get_litellm_service().check_connectivity()
    app.state.ready = readiness.healthy
    logger.info('startup_readiness healthy=%s checks=%s', readiness.healthy, readiness.checks)
    if not readiness.healthy:
        raise ServiceUnavailableError('Startup readiness validation failed. Check LiteLLM, vLLM, and model registry configuration.')
    try:
        yield
    finally:
        app.state.ready = False
        await get_http_client().aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version='3.0.0', lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(health_router)
    return app


app = create_app()
