"""Application configuration and dependency wiring for the Yn AI Setu Model Gateway."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
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

    # Defaults mirror .env.example so a bare `Settings()` (used by unit tests
    # and by any environment that hasn't set these explicitly) still resolves
    # to the same local vLLM-via-LiteLLM endpoint documented there.
    litellm_base_url: str = Field(default='http://127.0.0.1:4000/v1', alias='LITELLM_BASE_URL')
    litellm_api_key: SecretStr | None = Field(default=SecretStr('local'), alias='API_KEY')
    default_model: str = Field(default='qwen-erp', alias='DEFAULT_MODEL')
    request_timeout: float = Field(default=60.0, alias='REQUEST_TIMEOUT')
    max_retries: int = Field(default=3, alias='MAX_RETRIES')
    models_config_path: Path = Field(default=Path('configs/models.yaml'), alias='MODEL_REGISTRY_PATH')
    # Per-capability model profile config (planner/generate/security/general
    # chat routing) - see configs/model.yaml and ModelGatewayConfig below.
    model_gateway_config_path: Path = Field(default=Path('configs/model.yaml'), alias='MODEL_GATEWAY_CONFIG_PATH')
    litellm_timeout_seconds: float = Field(default=90.0, alias='LITELLM_TIMEOUT_SECONDS')

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


# ---------------------------------------------------------------------------
# Per-capability model profile configuration (configs/model.yaml).
#
# This backs the richer inference subsystem - planner, generation with tool
# grounding, security classification, and general chat (see
# app/services/dependencies.py) - which is the documented public API
# (README.md: "Public APIs: /planner, /generate, /security/classify,
# /chat/general"). It is distinct from the flat `ModelRegistry` above, which
# only backs the generic OpenAI-compatible /chat and /v1/models endpoints.
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str = 'litellm'
    api_base: str | None = None
    api_key: str | None = None
    deployment_name: str | None = None
    api_version: str | None = None
    organization: str | None = None
    timeout_seconds: float | None = None


class PromptsConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    planner: str | None = None
    generator: str = ''
    chat: str | None = None
    security: str | None = None


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.0
    planner_max_new_tokens: int = 128


class PlannerTarget(BaseModel):
    model_config = ConfigDict(extra='ignore')

    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    intent: str | None = None
    tool: str | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    response_type: str | None = None


# Flat, pre-nested-provider fields still accepted on a model profile for
# backward compatibility (README.md: "Legacy flat provider fields (base_url,
# api_key, api_version, organization, deployment_name) are still normalized
# into the new nested provider config").
_LEGACY_PROVIDER_FIELDS = ('base_url', 'api_base', 'api_key', 'deployment_name', 'api_version', 'organization', 'timeout_seconds')


class ModelProfileConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    model_name: str = ''
    base_model: str = ''
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    adapter_enabled: bool = False
    default_adapter: str | None = None
    adapters_root: str = 'adapters'
    device: str = 'remote'
    dtype: str = 'remote'
    trust_remote_code: bool = False
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    planner_targets: list[PlannerTarget] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_provider_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        provider = data.get('provider')
        if isinstance(provider, str):
            data['provider'] = {
                'type': provider,
                'api_base': data.pop('base_url', None) or data.pop('api_base', None),
                'api_key': data.pop('api_key', None),
                'deployment_name': data.pop('deployment_name', None),
                'api_version': data.pop('api_version', None),
                'organization': data.pop('organization', None),
                'timeout_seconds': data.pop('timeout_seconds', None),
            }
        elif provider is None and any(key in data for key in _LEGACY_PROVIDER_FIELDS):
            data['provider'] = {
                'type': data.pop('provider_type', 'litellm'),
                'api_base': data.pop('base_url', None) or data.pop('api_base', None),
                'api_key': data.pop('api_key', None),
                'deployment_name': data.pop('deployment_name', None),
                'api_version': data.pop('api_version', None),
                'organization': data.pop('organization', None),
                'timeout_seconds': data.pop('timeout_seconds', None),
            }
        return data

    @property
    def provider_type(self) -> str:
        return self.provider.type

    @property
    def provider_config(self) -> ProviderConfig:
        return self.provider


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    planner_model: str = 'erp'
    generate_model: str = 'erp'
    security_model: str = 'general'
    general_chat_model: str = 'general'


class ModelGatewayConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    models: dict[str, ModelProfileConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    def get_profile(self, profile_name: str) -> ModelProfileConfig:
        from app.core.exceptions import ResourceNotFoundError

        try:
            return self.models[profile_name]
        except KeyError as exc:
            available = ', '.join(sorted(self.models)) or 'none'
            raise ResourceNotFoundError(f'Model profile "{profile_name}" is not registered. Available: {available}.') from exc


_ENV_INTERPOLATION_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}')


def _interpolate_env(value: Any) -> Any:
    """Resolve `${ENV_VAR:default}` placeholders against the process environment."""

    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else '')

        return _ENV_INTERPOLATION_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: _interpolate_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


def load_model_gateway_config(path: Path) -> ModelGatewayConfig:
    """Load and env-interpolate configs/model.yaml into a ModelGatewayConfig."""

    from app.core.exceptions import ServiceUnavailableError

    if not path.exists():
        raise ServiceUnavailableError(f'Model gateway config file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    interpolated = _interpolate_env(raw)
    return ModelGatewayConfig.model_validate(interpolated)


@lru_cache(maxsize=1)
def get_model_gateway_config() -> ModelGatewayConfig:
    """Return the cached per-capability model profile configuration."""

    return load_model_gateway_config(get_settings().model_gateway_config_path)
