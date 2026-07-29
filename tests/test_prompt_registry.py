import pytest

from app.config.settings import ModelGatewayConfig
from app.registry.prompt_registry import PromptRegistry


def test_prompt_registry_returns_descriptor() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'general': {
                    'model_name': 'qwen-general',
                    'provider': 'openai',
                    'base_url': 'http://localhost:8000/v1',
                    'api_key': 'local',
                    'prompts': {'chat': 'hello'}
                }
            }
        }
    )
    profile = config.get_profile('general')
    descriptor = PromptRegistry().get('general', 'chat', profile)
    assert descriptor.content == 'hello'
    assert descriptor.prompt_type == 'chat'
