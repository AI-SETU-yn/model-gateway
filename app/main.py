from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.routers.inference import router as inference_router
from app.api.routers.management import router as management_router
from app.config.settings import get_model_config, get_settings
from app.exceptions.errors import AppException
from app.exceptions.handlers import app_exception_handler, request_validation_exception_handler, unhandled_exception_handler
from app.middleware.request_context import RequestContextMiddleware
from app.services.dependencies import get_model_loader
from app.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = get_model_config()
    configure_logging(settings)
    app.state.is_ready = False
    model_loader = get_model_loader()
    await model_loader.initialize(preload_default_adapter=bool(config.default_adapter))
    app.state.is_ready = settings.ready_on_startup or model_loader.base_model_loaded
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
