from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config.settings import ModelProfileConfig
from app.exceptions.errors import AppException


@dataclass(frozen=True)
class PromptDescriptor:
    profile: str
    prompt_type: str
    version: str
    content: str
    metadata: dict[str, str]


class PromptRegistry:
    _DEFAULT_VERSION = 'v1'

    def get(self, profile_name: str, prompt_type: str, profile: ModelProfileConfig) -> PromptDescriptor:
        return self._get_cached(
            profile_name,
            prompt_type,
            profile.prompts.planner,
            profile.prompts.generator,
            profile.prompts.chat,
            profile.prompts.security,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_cached(
        profile_name: str,
        prompt_type: str,
        planner: str | None,
        generator: str,
        chat: str | None,
        security: str | None,
    ) -> PromptDescriptor:
        mapping = {
            'planner': planner,
            'generator': generator,
            'chat': chat or generator,
            'security': security,
        }
        content = mapping.get(prompt_type)
        if not content:
            raise AppException(
                f'Prompt "{prompt_type}" is not configured for profile "{profile_name}".',
                code='INVALID_CONFIG',
                status_code=500,
            )
        return PromptDescriptor(
            profile=profile_name,
            prompt_type=prompt_type,
            version=PromptRegistry._DEFAULT_VERSION,
            content=content.strip(),
            metadata={'profile': profile_name, 'prompt_type': prompt_type, 'version': PromptRegistry._DEFAULT_VERSION},
        )
