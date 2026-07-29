import pytest

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import AppException
from app.registry.model_registry import ModelRegistry


def test_model_registry_validates_provider() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'x',
                    'provider': 'bad-provider',
                    'base_url': 'http://localhost:8000/v1',
                    'api_key': 'local',
                }
            }
        }
    )
    with pytest.raises(AppException):
        ModelRegistry(config)
