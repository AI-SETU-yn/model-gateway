"""Application configuration and dependency wiring for the Yn AI Setu Model Gateway."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.model_registry import ModelRegistry


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )

    app_name: str = 'yn-ai-setu-model-gateway'
    environment: Literal['local', 'dev', 'stage', 'prod', 'test'] = 'local'
    host: str = '0.0.0.0'
    port: int = 9000
    log_level: str = 'INFO'

    litellm_base_url: str = Field(alias='LITELLM_BASE_URL')
    litellm_api_key: SecretStr | None = Field(default=None, alias='API_KEY')
    default_model: str = Field(alias='DEFAULT_MODEL')
    request_timeout: float = Field(alias='REQUEST_TIMEOUT')
    max_retries: int = Field(alias='MAX_RETRIES')
    models_config_path: Path = Field(alias='MODEL_REGISTRY_PATH')

    @field_validator('litellm_base_url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip('/')
        if not normalized:
            raise ValueError('LITELLM_BASE_URL must not be empty.')
        if not normalized.startswith(('http://', 'https://')):
            raise ValueError('LITELLM_BASE_URL must start with http:// or https://.')
        return normalized

    @field_validator('default_model')
    @classmethod
    def validate_default_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError('DEFAULT_MODEL must not be empty.')
        return normalized

    @field_validator('request_timeout')
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError('REQUEST_TIMEOUT must be greater than zero.')
        return value

    @field_validator('max_retries')
    @classmethod
    def validate_retries(cls, value: int) -> int:
        if value < 1:
            raise ValueError('MAX_RETRIES must be at least 1.')
        return value

    @field_validator('models_config_path')
    @classmethod
    def validate_models_config_path(cls, value: Path) -> Path:
        normalized = Path(str(value)).expanduser()
        if not str(normalized).strip():
            raise ValueError('MODEL_REGISTRY_PATH must not be empty.')
        return normalized

    @property
    def api_key(self) -> str | None:
        return self.litellm_api_key.get_secret_value() if self.litellm_api_key else None

    @property
    def models_url(self) -> str:
        base = self.litellm_base_url
        return f'{base}/models' if base.endswith('/v1') else f'{base}/v1/models'

    @property
    def health_url(self) -> str:
        return f'{self.litellm_base_url}/health'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the cached YAML-backed model registry."""

    settings = get_settings()
    return ModelRegistry(settings.models_config_path)


@lru_cache(maxsize=1)
def get_http_client() -> httpx.AsyncClient:
    """Return a singleton outbound async HTTP client."""

    settings = get_settings()
    return httpx.AsyncClient(timeout=settings.request_timeout)


@lru_cache(maxsize=1)
def get_litellm_service():
    """Return the cached LiteLLM-backed inference service."""

    from app.services.litellm_service import LiteLLMService

    return LiteLLMService(get_settings(), get_model_registry(), get_http_client())


@lru_cache(maxsize=1)
def get_chat_service():
    """Return the cached chat business service."""

    from app.services.chat_service import ChatService

    return ChatService(get_litellm_service(), get_settings(), get_model_registry())
