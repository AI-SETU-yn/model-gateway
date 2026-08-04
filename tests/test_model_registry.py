import pytest

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import AppException, MissingProviderApiKey, MissingProviderEndpoint
from app.registry.model_registry import ModelRegistry


def test_model_registry_validates_provider() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'x',
                    'base_model': 'y',
                    'provider': {'type': 'bad-provider'},
                }
            }
        }
    )
    with pytest.raises(AppException):
        ModelRegistry(config)


def test_model_registry_requires_endpoint_for_vllm() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'x',
                    'base_model': 'y',
                    'provider': {'type': 'vllm', 'api_key': 'local'},
                }
            }
        }
    )
    with pytest.raises(MissingProviderEndpoint):
        ModelRegistry(config)


def test_model_registry_requires_api_key_for_vllm() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'x',
                    'base_model': 'y',
                    'provider': {'type': 'vllm', 'api_base': 'http://localhost:4000/v1'},
                }
            }
        }
    )
    with pytest.raises(MissingProviderApiKey):
        ModelRegistry(config)
