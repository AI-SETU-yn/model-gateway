from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

from app.config.settings import ModelGatewayConfig, ModelProfileConfig
from app.exceptions.errors import AppException

SUPPORTED_PROVIDERS = frozenset({'openai', 'azure', 'ollama', 'litellm'})


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
            if profile.provider not in SUPPORTED_PROVIDERS:
                raise AppException(f'Unsupported provider "{profile.provider}" for profile "{name}".', code='INVALID_CONFIG', status_code=500)
            if not profile.model_name or not profile.base_url or not profile.api_key:
                raise AppException(f'Profile "{name}" is missing required model connection fields.', code='INVALID_CONFIG', status_code=500)

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
