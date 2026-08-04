import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_http_client, get_settings


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    get_http_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_http_client.cache_clear()


def test_settings_require_phase3_environment_variables(monkeypatch):
    for key in ['LITELLM_BASE_URL', 'DEFAULT_MODEL', 'REQUEST_TIMEOUT', 'MAX_RETRIES', 'MODEL_REGISTRY_PATH']:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_load_required_environment_variables(monkeypatch, tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text('models: {qwen-erp: {provider: litellm, backend: vllm, endpoint: http://127.0.0.1:4000/v1, supports_streaming: true, default: true}}', encoding='utf-8')
    monkeypatch.setenv('LITELLM_BASE_URL', 'http://127.0.0.1:4000/v1')
    monkeypatch.setenv('DEFAULT_MODEL', 'qwen-erp')
    monkeypatch.setenv('REQUEST_TIMEOUT', '60')
    monkeypatch.setenv('MAX_RETRIES', '3')
    monkeypatch.setenv('MODEL_REGISTRY_PATH', str(registry_path))

    settings = Settings()

    assert settings.litellm_base_url == 'http://127.0.0.1:4000/v1'
    assert settings.default_model == 'qwen-erp'
    assert str(settings.models_config_path) == str(registry_path)


def test_http_client_is_singleton(monkeypatch, tmp_path):
    registry_path = tmp_path / 'models.yaml'
    registry_path.write_text('models: {qwen-erp: {provider: litellm, backend: vllm, endpoint: http://127.0.0.1:4000/v1, supports_streaming: true, default: true}}', encoding='utf-8')
    monkeypatch.setenv('LITELLM_BASE_URL', 'http://127.0.0.1:4000/v1')
    monkeypatch.setenv('DEFAULT_MODEL', 'qwen-erp')
    monkeypatch.setenv('REQUEST_TIMEOUT', '60')
    monkeypatch.setenv('MAX_RETRIES', '3')
    monkeypatch.setenv('MODEL_REGISTRY_PATH', str(registry_path))

    client_a = get_http_client()
    client_b = get_http_client()

    assert client_a is client_b
