import pytest

from app.config.settings import ModelGatewayConfig
from app.registry.prompt_registry import PromptRegistry


def test_prompt_registry_returns_descriptor() -> None:
    config = ModelGatewayConfig.model_validate(
        {
            'models': {
                'general': {
                    'model_name': 'qwen-general',
                    'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'provider': 'transformers',
                    'prompts': {'chat': 'hello'}
                }
            },
            'routing': {
                'planner_model': 'general',
                'generate_model': 'general',
                'security_model': 'general',
                'general_chat_model': 'general',
            },
        }
    )
    profile = config.get_profile('general')
    descriptor = PromptRegistry().get('general', 'chat', profile)
    assert descriptor.content == 'hello'
    assert descriptor.prompt_type == 'chat'
