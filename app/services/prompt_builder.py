"""Prompt preparation for response generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.schemas.inference import GenerateMessage, GenerateRequest, GenerationPolicy


@dataclass(frozen=True)
class PromptBundle:
    """Model-agnostic chat messages ready for template rendering."""

    messages: list[dict[str, str]]
    source: str


class PromptBuilder:
    """Build model-facing chat messages from legacy and structured requests."""

    _TRANSPORT_METADATA_KEYS = frozenset(
        {
            'tool_name',
            'server',
            'status',
            'success',
            'response_type',
            'error',
            'registry_lookup_latency_ms',
            'tool_execution_latency_ms',
        }
    )

    def __init__(self, default_system_prompt: str) -> None:
        self._default_system_prompt = default_system_prompt.strip()

    def build(self, request: GenerateRequest) -> PromptBundle:
        if request.messages:
            messages = self._normalize_messages(request.messages)
            source = 'structured_messages'
        else:
            messages = [{'role': 'user', 'content': request.prompt or ''}]
            source = 'legacy_prompt'

        system_prompt = self._build_system_prompt(request.generation_policy, request.response_type)
        messages = self._ensure_system_message(messages, system_prompt)

        if request.tool_result is not None:
            messages.append(
                {
                    'role': 'tool',
                    'content': self._serialize_tool_result(request.tool_result),
                }
            )
            source = f'{source}+tool_result'

        return PromptBundle(messages=messages, source=source)

    def _build_system_prompt(self, policy: GenerationPolicy | None, response_type: str | None) -> str:
        extras: list[str] = []
        if response_type:
            extras.append(f'Target response type: {response_type}.')

        if policy is None:
            if not extras:
                return self._default_system_prompt
            return f"{self._default_system_prompt}\n\n" + '\n'.join(extras)

        rules: list[str] = extras
        if policy.use_tool_results_only:
            rules.append('Use only supplied tool results for enterprise facts.')
        if policy.never_invent_business_data:
            rules.append('Never invent business data or identifiers.')
        if policy.never_ask_for_present_data:
            rules.append('Never ask for information already present in supplied data.')
        if policy.ignore_orchestration_metadata:
            rules.append('Ignore orchestration metadata and debug fields.')
        if policy.concise:
            rules.append('Keep the response concise and business-focused.')
        if not rules:
            return self._default_system_prompt
        return f"{self._default_system_prompt}\n\nAdditional generation policy:\n" + '\n'.join(f'- {rule}' for rule in rules)

    @staticmethod
    def _normalize_messages(messages: list[GenerateMessage]) -> list[dict[str, str]]:
        return [{'role': message.role, 'content': message.content} for message in messages]

    @staticmethod
    def _ensure_system_message(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
        if not messages:
            return [{'role': 'system', 'content': system_prompt}]
        if messages[0]['role'] == 'system':
            first = dict(messages[0])
            first['content'] = f"{system_prompt}\n\n{first['content']}".strip()
            return [first, *messages[1:]]
        return [{'role': 'system', 'content': system_prompt}, *messages]

    @classmethod
    def _serialize_tool_result(cls, tool_result: Any) -> str:
        normalized = cls._extract_business_payload(tool_result)
        normalized = cls._normalize_json_strings(normalized)
        if isinstance(normalized, str):
            return normalized
        return json.dumps(normalized, ensure_ascii=False, default=str, indent=2)

    @classmethod
    def _extract_business_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if 'data' in value and cls._TRANSPORT_METADATA_KEYS.intersection(value):
                return cls._extract_business_payload(value['data'])
            content = value.get('content')
            if isinstance(content, list):
                extracted = cls._extract_mcp_text_content(content)
                if extracted is not None:
                    return cls._extract_business_payload(extracted)
        return value

    @classmethod
    def _extract_mcp_text_content(cls, content: list[Any]) -> Any | None:
        text_items: list[Any] = []
        for item in content:
            if not isinstance(item, dict) or 'text' not in item:
                return None
            text_items.append(cls._normalize_json_strings(item['text']))
        if not text_items:
            return None
        return text_items[0] if len(text_items) == 1 else text_items

    @classmethod
    def _normalize_json_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.startswith(('{', '[')):
                try:
                    return cls._normalize_json_strings(json.loads(candidate))
                except json.JSONDecodeError:
                    return value
            return value
        if isinstance(value, list):
            return [cls._normalize_json_strings(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._normalize_json_strings(item) for key, item in value.items()}
        return value
