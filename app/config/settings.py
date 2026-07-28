"""Configuration loaders for the model gateway."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_GENERATE_SYSTEM_PROMPT = """You are the YN Setu enterprise assistant.
Answer using only supplied enterprise tool results when tool results are available.
Never invent business data, names, identifiers, dates, counts, or statuses.
Never ask for information that is already present in the supplied data.
Ignore orchestration metadata such as planner intents, tool names, server names, trace IDs, latency, and debug fields.
Produce concise, professional enterprise responses."""


class GenerationConfig(BaseModel):
    """Text generation settings."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.05
    planner_max_new_tokens: int = 256
    use_chat_template: bool = True


class ModelGatewayConfig(BaseModel):
    """YAML-backed gateway configuration."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    base_model: str
    default_adapter: str
    adapters_root: Path
    device: str = 'auto'
    dtype: str = 'auto'
    trust_remote_code: bool = True
    planner_system_prompt: str
    generate_system_prompt: str = DEFAULT_GENERATE_SYSTEM_PROMPT
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> 'ModelGatewayConfig':
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        return cls.model_validate(payload)


class Settings(BaseSettings):
    """Environment-driven runtime settings."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='MODEL_GATEWAY_',
        case_sensitive=False,
        extra='ignore',
    )

    app_name: str = 'yn-ai-setu-model-gateway'
    environment: Literal['local', 'dev', 'stage', 'prod', 'test'] = 'local'
    host: str = '0.0.0.0'
    port: int = 9000
    log_level: str = 'INFO'
    ready_on_startup: bool = True
    config_path: Path = Path('configs/model.yaml')
    default_adapter: str | None = None
    device: str | None = None
    dtype: str | None = None
    preload_default_adapter: bool = False
    max_concurrent_requests: int = 1
    metrics_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_model_config() -> ModelGatewayConfig:
    settings = get_settings()
    path = settings.config_path
    config = ModelGatewayConfig.from_yaml(path)
    update_payload: dict[str, object] = {}
    if settings.default_adapter:
        update_payload['default_adapter'] = settings.default_adapter
    if settings.device:
        update_payload['device'] = settings.device
    if settings.dtype:
        update_payload['dtype'] = settings.dtype
    if update_payload:
        config = config.model_copy(update=update_payload)
    return config
