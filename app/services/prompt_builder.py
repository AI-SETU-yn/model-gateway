"""Prompt preparation for response generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from app.schemas.inference import GenerateMessage, GenerateRequest, GenerationPolicy


@dataclass(frozen=True)
class PromptBundle:
    """Model-agnostic chat messages ready for template rendering."""

    messages: list[dict[str, str]]
    source: str
    preprocessing_ms: float
    building_ms: float


class PromptBuilder:
    """Build model-facing chat messages from structured response requests."""

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
        preprocessing_started = time.perf_counter()
        tool_result = request.tool_result
        if request.messages:
            messages = self._normalize_messages(request.messages)
            source = 'structured_messages'
        else:
            messages = []
            source = 'structured_tool_result'

        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
        building_started = time.perf_counter()
        if tool_result is not None and source == 'structured_messages':
            source = 'structured_messages+tool_result'

        if request.messages and tool_result is None:
            source = 'structured_messages'

        system_prompt = self._build_system_prompt(request.generation_policy, request.response_type)
        messages = self._ensure_system_message(messages, system_prompt)

        response_context = self._response_context(request)
        if response_context:
            messages.append(
                {
                    'role': 'tool',
                    'content': json.dumps({'response_context': response_context}, ensure_ascii=False, default=str, indent=2),
                }
            )

        if tool_result is not None:
            messages.append(
                {
                    'role': 'tool',
                    'content': self._serialize_tool_result(request),
                }
            )

        building_ms = (time.perf_counter() - building_started) * 1000
        return PromptBundle(
            messages=messages,
            source=source,
            preprocessing_ms=round(preprocessing_ms, 2),
            building_ms=round(building_ms, 2),
        )

    def _build_system_prompt(self, policy: GenerationPolicy | None, response_type: str | None) -> str:
        extras: list[str] = []
        if response_type:
            extras.append(f'Target response type: {response_type}.')
            extras.extend(self._response_type_rules(response_type))

        if policy is None:
            if not extras:
                return self._default_system_prompt
            return f"{self._default_system_prompt}\n\n" + '\n'.join(extras)

        rules: list[str] = extras
        if policy.grounded:
            rules.append('Ground the answer in the supplied structured context and tool results.')
        if policy.hallucination:
            rules.append(f'Hallucination policy: {policy.hallucination}.')
        if policy.output_format:
            rules.append(f'Preferred output format: {policy.output_format}.')
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
    def _response_type_rules(response_type: str) -> list[str]:
        normalized = response_type.strip().casefold()
        if normalized in {'enterprise', 'multi_tool', 'current_info'}:
            return [
                'Create the final user-facing answer from the supplied data only.',
                'Do not expose raw orchestration metadata unless the user explicitly asks for it.',
            ]
        if normalized == 'clarification':
            return [
                'Ask one concise question that helps collect the missing required parameter values.',
                'If options are supplied, present them clearly without inventing additional options.',
            ]
        if normalized == 'tool_failure':
            return [
                'Explain that the requested data could not be retrieved using the supplied failure details.',
                'Do not invent replacement enterprise data.',
            ]
        if normalized == 'planner_failure':
            return [
                'Explain that the request could not be matched to an available action.',
                'Ask the user to rephrase or provide the missing detail.',
            ]
        if normalized == 'empty_result':
            return [
                'Explain that no matching records were returned.',
                'Do not invent records.',
            ]
        if normalized == 'validation_retry':
            return [
                'Rewrite the previous answer so it is strictly grounded in the supplied data.',
                'Remove any unsupported claims.',
            ]
        return []

    @staticmethod
    def _normalize_messages(messages: list[GenerateMessage]) -> list[dict[str, str]]:
        return [{'role': message.role, 'content': message.content} for message in messages]

    @staticmethod
    def _response_context(request: GenerateRequest) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if request.conversation is not None:
            context['conversation'] = request.conversation.model_dump(by_alias=True, exclude_none=True)
        if request.missing_parameters:
            context['missingParameters'] = request.missing_parameters
        if request.metadata:
            context['metadata'] = request.metadata
        return context

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
    def _serialize_tool_result(cls, request: GenerateRequest) -> str:
        normalized = cls._tool_result_for_prompt(request)
        normalized = cls._normalize_json_strings(normalized)
        if isinstance(normalized, str):
            return normalized
        return json.dumps(normalized, ensure_ascii=False, default=str, indent=2)

    @classmethod
    def _tool_result_for_prompt(cls, request: GenerateRequest) -> Any:
        tool_result = request.tool_result
        if not cls._is_multi_tool_request(request) or not isinstance(tool_result, dict):
            return cls._extract_payload(tool_result)

        steps = tool_result.get('steps')
        if not isinstance(steps, list):
            return cls._extract_payload(tool_result)

        visible_step_ids = cls._visible_step_ids(request)
        if not visible_step_ids:
            return cls._extract_payload(tool_result.get('data', tool_result))

        step_by_id = {
            step.get('step_id'): step
            for step in steps
            if isinstance(step, dict) and isinstance(step.get('step_id'), str)
        }
        visible_results = []
        for step_id in visible_step_ids:
            step = step_by_id.get(step_id)
            if not isinstance(step, dict):
                continue
            result = step.get('result')
            if result is not None:
                visible_results.append(cls._extract_payload(result))

        if not visible_results:
            return cls._extract_payload(tool_result.get('data', tool_result))
        if len(visible_results) == 1:
            return visible_results[0]
        return {'results': visible_results}

    @staticmethod
    def _is_multi_tool_request(request: GenerateRequest) -> bool:
        return (request.response_type or '').strip().casefold() == 'multi_tool'

    @staticmethod
    def _visible_step_ids(request: GenerateRequest) -> list[str]:
        if request.conversation is None:
            return []
        visible_step_ids = []
        for index, step in enumerate(request.conversation.execution_plan, start=1):
            if step.get('visible_in_response') is False or step.get('visibleInResponse') is False:
                continue
            step_id = step.get('step_id') or step.get('stepId') or f'step_{index}'
            if isinstance(step_id, str):
                visible_step_ids.append(step_id)
        return visible_step_ids

    @classmethod
    def _extract_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if 'data' in value and cls._TRANSPORT_METADATA_KEYS.intersection(value):
                return cls._extract_payload(value['data'])
            content = value.get('content')
            if isinstance(content, list):
                extracted = cls._extract_mcp_text_content(content)
                if extracted is not None:
                    return cls._extract_payload(extracted)
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
