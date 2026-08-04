"""YAML-backed model registry for gateway model resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.exceptions import BadRequestError, ResourceNotFoundError, ServiceUnavailableError


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""



def _construct_unique_mapping(loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ServiceUnavailableError(f'Duplicate model registry key detected: {key}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


class RegisteredModel(BaseModel):
    """Single model profile registered with the gateway."""

    model_config = ConfigDict(extra='forbid')

    provider: str
    backend: str
    endpoint: str
    supports_streaming: bool = Field(alias='supports_streaming')
    default: bool = False

    @model_validator(mode='after')
    def validate_fields(self) -> 'RegisteredModel':
        self.provider = self.provider.strip()
        self.backend = self.backend.strip()
        self.endpoint = self.endpoint.strip().rstrip('/')
        if not self.provider:
            raise ValueError('provider must not be empty.')
        if not self.backend:
            raise ValueError('backend must not be empty.')
        if not self.endpoint.startswith(('http://', 'https://')):
            raise ValueError('endpoint must start with http:// or https://.')
        return self


class ModelRegistryPayload(BaseModel):
    """Full YAML registry document."""

    model_config = ConfigDict(extra='forbid')

    models: dict[str, RegisteredModel]

    @model_validator(mode='after')
    def validate_defaults(self) -> 'ModelRegistryPayload':
        defaults = [name for name, model in self.models.items() if model.default]
        if not self.models:
            raise ValueError('models registry must contain at least one model.')
        if len(defaults) > 1:
            raise ValueError(f'Only one default model is allowed, found: {", ".join(defaults)}')
        return self


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
            raw = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueKeyLoader) or {}
            return ModelRegistryPayload.model_validate(raw)
        except ServiceUnavailableError:
            raise
        except (yaml.YAMLError, ValidationError, ValueError) as exc:
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
