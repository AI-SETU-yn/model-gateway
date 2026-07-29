from contextlib import asynccontextmanager
from pathlib import Path
import os

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


GPT_REFERENCE_TERMS = ('sshleifer/tiny-gpt2', 'tiny-gpt2', 'distilgpt2', 'gpt2', 'sshleifer')
ENV_VERIFICATION_KEYS = ('BASE_MODEL', 'MODEL_NAME', 'MODEL_ID', 'HF_MODEL', 'BASE_MODEL_NAME', 'HUGGINGFACE_MODEL')


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path(settings) -> Path:
    return (Path.cwd() / settings.config_path).resolve()


def _adapter_location(config) -> Path:
    return (_project_root() / config.adapters_root / config.default_adapter).resolve()


def _log_environment_verification() -> None:
    for key in ENV_VERIFICATION_KEYS:
        value = os.environ.get(key)
        if value is None:
            print(f'Environment verification: {key}=Not Set')
        else:
            print(f'Environment verification: {key}={value}')


def _scan_repository_for_gpt_references(root: Path) -> None:
    matches: list[str] = []
    skipped_dirs = {'.git', '__pycache__', 'myenv', '.pytest_cache'}
    for path in root.rglob('*'):
        if any(part in skipped_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            lowered = line.lower()
            if any(term in lowered for term in GPT_REFERENCE_TERMS):
                matches.append(f'{path.relative_to(root)}:{line_number}: {line.strip()}')
    if matches:
        print('Repository GPT-2 reference scan results:')
        for match in matches:
            print(match)
        return
    print('No GPT-2 references found in current repository.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = get_model_config()
    configure_logging(settings)
    app.state.is_ready = False

    project_root = _project_root()
    loaded_config_path = _config_path(settings)
    print(f'Loaded configuration: {loaded_config_path}')
    print(f'Current working directory: {Path.cwd().resolve()}')
    print(f'Application root: {project_root}')
    print(f'Location of model.yaml: {loaded_config_path}')
    print(f'Location of adapter: {_adapter_location(config)}')
    _log_environment_verification()
    _scan_repository_for_gpt_references(project_root)

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
