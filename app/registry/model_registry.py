from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

from app.config.settings import ModelGatewayConfig, ModelProfileConfig
from app.exceptions.errors import AppException, MissingProviderApiKey, MissingProviderEndpoint

SUPPORTED_PROVIDERS = frozenset({
    'litellm',
    'vllm',
    'openai',
    'azure',
    'azure_openai',
    'ollama',
    'huggingface',
    'huggingface_endpoint',
    'together_ai',
    'togetherai',
    'groq',
    'openrouter',
})

_PROVIDER_ENDPOINT_REQUIRED = frozenset({'litellm', 'vllm', 'ollama', 'huggingface_endpoint', 'azure', 'azure_openai'})
_PROVIDER_API_KEY_REQUIRED = frozenset({'openai', 'azure', 'azure_openai', 'huggingface', 'huggingface_endpoint', 'together_ai', 'togetherai', 'groq', 'openrouter', 'litellm', 'vllm'})


class ModelRegistry:
    def __init__(self, config: ModelGatewayConfig) -> None:
        self._config = config
        self._profiles = OrderedDict(sorted(config.models.items(), key=lambda item: item[0]))
        self._validate()

    def _validate(self) -> None:
        aliases = set()
        for name, profile in self._profiles.items():
            if name in aliases:
                raise AppException(f'Duplicate model profile alias: {name}', code='INVALID_CONFIG', status_code=500)
            aliases.add(name)
            provider_type = profile.provider_type
            if provider_type not in SUPPORTED_PROVIDERS:
                raise AppException(f'Unsupported provider "{provider_type}" for profile "{name}".', code='INVALID_CONFIG', status_code=500)
            if not profile.model_name or not profile.base_model:
                raise AppException(f'Profile "{name}" is missing required model fields.', code='INVALID_CONFIG', status_code=500)
            provider = profile.provider_config
            if provider_type in _PROVIDER_ENDPOINT_REQUIRED and not (provider.api_base or '').strip():
                raise MissingProviderEndpoint(f'Provider "{provider_type}" for profile "{name}" requires api_base.')
            if provider_type in _PROVIDER_API_KEY_REQUIRED and not (provider.api_key or '').strip():
                raise MissingProviderApiKey(f'Provider "{provider_type}" for profile "{name}" requires api_key.')
            if provider_type in {'azure', 'azure_openai'} and not (provider.deployment_name or profile.model_name or '').strip():
                raise AppException(f'Provider "{provider_type}" for profile "{name}" requires deployment_name.', code='INVALID_CONFIG', status_code=500)

    @lru_cache(maxsize=64)
    def get(self, profile_name: str) -> ModelProfileConfig:
        try:
            return self._profiles[profile_name]
        except KeyError as exc:
            available = ', '.join(self._profiles) or 'none'
            raise AppException(f'Unknown model profile "{profile_name}". Available: {available}.', code='INVALID_CONFIG', status_code=500) from exc

    def route(self, route_name: str) -> tuple[str, ModelProfileConfig]:
        profile_name = getattr(self._config.routing, route_name)
        return profile_name, self.get(profile_name)

    def planner_profile(self) -> tuple[str, ModelProfileConfig]:
        return self.route('planner_model')

    def generation_profile(self) -> tuple[str, ModelProfileConfig]:
        return self.route('generate_model')

    def security_profile(self) -> tuple[str, ModelProfileConfig]:
        return self.route('security_model')

    def general_chat_profile(self) -> tuple[str, ModelProfileConfig]:
        return self.route('general_chat_model')

    def all_profiles(self) -> dict[str, ModelProfileConfig]:
        return dict(self._profiles)
