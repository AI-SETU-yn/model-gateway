"""Configuration loaders for the self-hosted Transformers-backed model gateway."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GENERATE_SYSTEM_PROMPT = """You are the YN Setu enterprise assistant.
Answer using only supplied enterprise tool results when tool results are available.
You own all user-facing wording for enterprise, general, current information, clarification, failure, empty-result, and multi-tool responses.
Never invent business data, names, identifiers, dates, counts, or statuses.
Never ask for information that is already present in the supplied data.
Ignore orchestration metadata such as planner intents, tool names, server names, trace IDs, latency, and debug fields.
Produce concise, professional enterprise responses."""


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.05
    planner_max_new_tokens: int = 128


class ModelPromptsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    planner: str | None = None
    generator: str = DEFAULT_GENERATE_SYSTEM_PROMPT
    chat: str | None = None
    security: str | None = None


class ModelProfileConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    model_name: str
    base_model: str
    provider: str
    deployment_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    organization: str | None = None
    adapter_enabled: bool = False
    default_adapter: str | None = None
    adapters_root: str = 'adapters'
    device: str = 'auto'
    dtype: str = 'auto'
    trust_remote_code: bool = True
    prompts: ModelPromptsConfig = Field(default_factory=ModelPromptsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    planner_model: str
    generate_model: str
    security_model: str
    general_chat_model: str


class ModelGatewayConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    models: dict[str, ModelProfileConfig]
    routing: RoutingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> 'ModelGatewayConfig':
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
        return cls.model_validate(payload)

    def get_profile(self, name: str) -> ModelProfileConfig:
        try:
            return self.models[name]
        except KeyError as exc:
            available = ', '.join(sorted(self.models)) or 'none'
            raise ValueError(f'Unknown model profile "{name}". Available profiles: {available}.') from exc


class Settings(BaseSettings):
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
    max_concurrent_requests: int = 4
    metrics_enabled: bool = True
    health_timeout_seconds: float = 10.0
    litellm_timeout_seconds: float = 90.0
    litellm_max_retries: int = 2
    litellm_circuit_failure_threshold: int = 5
    litellm_circuit_recovery_timeout_seconds: float = 30.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_model_config() -> ModelGatewayConfig:
    settings = get_settings()
    return ModelGatewayConfig.from_yaml(settings.config_path)
