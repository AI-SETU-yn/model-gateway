import pytest

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import AppException
from app.registry.model_registry import ModelRegistry


def test_model_registry_requires_provider() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'erp': {
                    'model_name': 'x',
                    'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'provider': ' ',
                }
            },
            'routing': {
                'planner_model': 'erp',
                'generate_model': 'erp',
                'security_model': 'erp',
                'general_chat_model': 'erp',
            },
        }
    )
    with pytest.raises(AppException):
        ModelRegistry(config)
