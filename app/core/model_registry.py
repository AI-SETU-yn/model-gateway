"""YAML-backed model registry for gateway model resolution."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import BadRequestError, ResourceNotFoundError, ServiceUnavailableError


class RegisteredModel(BaseModel):
    """Single model profile registered with the gateway."""

    model_config = ConfigDict(extra='forbid')

    provider: str
    backend: str
    endpoint: str
    supports_streaming: bool = Field(alias='supports_streaming')
    default: bool = False


class ModelRegistryPayload(BaseModel):
    """Full YAML registry document."""

    model_config = ConfigDict(extra='forbid')

    models: dict[str, RegisteredModel]


class ModelRegistry:
    """Resolve models from `configs/models.yaml` without hardcoding names."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._payload = self._load(path)

    @staticmethod
    def _load(path: Path) -> ModelRegistryPayload:
        if not path.exists():
            raise ServiceUnavailableError(f'Model registry file not found: {path}')
        try:
            raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            return ModelRegistryPayload.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as exc:
            raise ServiceUnavailableError(f'Invalid model registry configuration: {exc}') from exc

    @property
    def path(self) -> Path:
        return self._path

    def get(self, model_name: str) -> RegisteredModel:
        try:
            return self._payload.models[model_name]
        except KeyError as exc:
            raise ResourceNotFoundError(f'Model "{model_name}" is not registered in models.yaml.') from exc

    def resolve(self, requested_model: str | None) -> tuple[str, RegisteredModel]:
        if requested_model:
            normalized = requested_model.strip()
            if not normalized:
                raise BadRequestError('Model name must not be empty when provided.')
            return normalized, self.get(normalized)
        default_entry = self.default_model()
        if default_entry is None:
            raise ServiceUnavailableError('No default model is configured in models.yaml.')
        return default_entry

    def default_model(self) -> tuple[str, RegisteredModel] | None:
        for name, model in self._payload.models.items():
            if model.default:
                return name, model
        return None

    def list_models(self) -> dict[str, RegisteredModel]:
        return dict(self._payload.models)
