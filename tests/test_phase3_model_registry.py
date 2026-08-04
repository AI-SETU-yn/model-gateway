import pytest

from app.core.exceptions import ResourceNotFoundError, ServiceUnavailableError
from app.core.model_registry import ModelRegistry


def test_model_registry_resolves_default_model(tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text(
        'models:\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4000/v1\n    supports_streaming: true\n    default: true\n',
        encoding='utf-8',
    )

    registry = ModelRegistry(registry_path)
    name, profile = registry.resolve(None)

    assert name == 'qwen-erp'
    assert profile.default is True


def test_model_registry_rejects_duplicate_keys(tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text(
        'models:\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4000/v1\n    supports_streaming: true\n    default: true\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4001/v1\n    supports_streaming: false\n',
        encoding='utf-8',
    )

    with pytest.raises(ServiceUnavailableError, match='Duplicate model registry key detected'):
        ModelRegistry(registry_path)


def test_model_registry_rejects_multiple_defaults(tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text(
        'models:\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4000/v1\n    supports_streaming: true\n    default: true\n  qwen-general:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4001/v1\n    supports_streaming: true\n    default: true\n',
        encoding='utf-8',
    )

    with pytest.raises(ServiceUnavailableError, match='Only one default model is allowed'):
        ModelRegistry(registry_path)


def test_model_registry_raises_for_missing_model(tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text(
        'models:\n  qwen-erp:\n    provider: litellm\n    backend: vllm\n    endpoint: http://127.0.0.1:4000/v1\n    supports_streaming: true\n    default: true\n',
        encoding='utf-8',
    )

    registry = ModelRegistry(registry_path)

    with pytest.raises(ResourceNotFoundError):
        registry.get('missing-model')
