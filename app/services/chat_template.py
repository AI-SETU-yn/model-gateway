"""Chat-template rendering isolated from inference orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    used_chat_template: bool


class ChatTemplateHandler:
    """Render chat messages using the tokenizer's native template when available."""

    def render(self, tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], *, use_chat_template: bool) -> RenderedPrompt:
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return RenderedPrompt(text=str(rendered), used_chat_template=True)
            except Exception as exc:
                logger.warning('chat_template_render_failed', extra={'error_type': type(exc).__name__})

        return RenderedPrompt(text=self._fallback_render(messages), used_chat_template=False)

    @staticmethod
    def _fallback_render(messages: list[dict[str, str]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = message['role'].upper()
            rendered.append(f"{role}:\n{message['content']}")
        rendered.append('ASSISTANT:')
        return '\n\n'.join(rendered)
