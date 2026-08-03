"""Planner inference service."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config.settings import ModelProfileConfig
from app.exceptions.errors import PlannerResponseError
from app.registry.prompt_registry import PromptRegistry
from app.schemas.inference import ExecutionPlanStep, PlannerRequest, PlannerResponse
from app.services.inference import ERPInferenceService

logger = logging.getLogger(__name__)


class PlannerService:
    def __init__(
        self,
        profile_name: str,
        profile: ModelProfileConfig,
        prompt_registry: PromptRegistry,
        inference_service: ERPInferenceService,
    ) -> None:
        self._profile_name = profile_name
        self._profile = profile
        self._prompt_registry = prompt_registry
        self._inference_service = inference_service

    async def plan(self, request: PlannerRequest) -> PlannerResponse:
        prompt = self._build_prompt(request)
        try:
            raw_response, _, model_name = await self._inference_service.planner_generate(request.adapter, prompt)
            parsed = self._parse_json(raw_response)
        except Exception as exc:
            logger.exception('planner_model_failed_using_fallback', extra={'adapter': request.adapter, 'query': request.query})
            return self._fallback_plan(request, raw_response=f'fallback:{type(exc).__name__}', model_name=self._profile.model_name)

        domain = self._get_text(parsed, 'domain')
        entity = self._get_text(parsed, 'entity')
        operation = self._get_text(parsed, 'operation')
        service = self._get_text(parsed, 'service') or self._derive_service(request.adapter, entity, operation)
        intent = self._get_text(parsed, 'intent') or self._compose_intent(service, entity, operation)
        tool = self._get_text(parsed, 'tool')
        parameters = self._normalize_parameters(parsed.get('parameters'))
        requires_tool = self._get_bool(parsed, 'requiresTool')
        if requires_tool is None:
            requires_tool = self._get_bool(parsed, 'requires_tool')
        execution_plan = self._execution_plan(parsed.get('executionPlan') or parsed.get('execution_plan'))
        if execution_plan and requires_tool is not True:
            requires_tool = True
        response_type = self._get_text(parsed, 'responseType')
        confidence = self._get_float(parsed, 'confidence')
        return PlannerResponse(
            intent=intent,
            tool=tool,
            domain=domain,
            service=service,
            entity=entity,
            operation=operation,
            parameters=parameters,
            requiresTool=requires_tool,
            responseType=response_type,
            executionPlan=execution_plan,
            confidence=confidence,
            rawResponse=raw_response,
            adapter=request.adapter,
            model=model_name,
        )

    def _fallback_plan(self, request: PlannerRequest, *, raw_response: str, model_name: str) -> PlannerResponse:
        return PlannerResponse(
            intent='general.chat', tool=None, domain=None, service=None, entity=None, operation=None,
            parameters={}, requiresTool=False, responseType='text', executionPlan=[], confidence=0.0,
            rawResponse=raw_response, adapter=request.adapter, model=model_name,
        )

    def _build_prompt(self, request: PlannerRequest) -> str:
        if request.prompt:
            return request.prompt
        planner_prompt = self._prompt_registry.get(self._profile_name, 'planner', self._profile).content
        registry_section = ''
        if request.registry_context:
            registry_section = (
                '\nRegistry context supplied by AI Runtime:\n'
                f'{request.registry_context}\n'
            )
        if request.registry_metadata is not None:
            registry_section = (
                f'{registry_section}\nRegistry metadata supplied by AI Runtime:\n'
                f'{json.dumps(request.registry_metadata, ensure_ascii=False, default=str)}\n'
            )
        return (
            f"{planner_prompt}\n"
            f"{registry_section}"
            'Schema: {"domain": string|null, "service": string|null, "entity": string|null, '
            '"operation": string|null, "intent": string|null, "tool": string|null, '
            '"parameters": object, "requiresTool": boolean|null, "responseType": string|null, '
            '"executionPlan": [{"stepId": string, "domain": string|null, "service": string|null, '
            '"entity": string|null, "operation": string|null, "intent": string|null, '
            '"tool": string|null, "parameters": object, "dependsOn": [string], '
            '"parameterBindings": object}], "confidence": number|null}.\n'
            'Return only valid JSON. If you do not know an exact tool name, set "tool" to null.\n'
            f'User Query: {request.query}\n'
            'JSON:'
        )

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, object]:
        candidate = raw_response.strip()
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            start = candidate.find('{')
            if start != -1:
                try:
                    parsed, _ = decoder.raw_decode(candidate[start:])
                except json.JSONDecodeError as exc:
                    raise PlannerResponseError('Planner model did not return valid JSON.') from exc
            else:
                raise PlannerResponseError('Planner model did not return valid JSON.')
        if not isinstance(parsed, dict):
            raise PlannerResponseError('Planner model did not return a JSON object.')
        return parsed

    @staticmethod
    def _get_text(payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return str(value)

    @staticmethod
    def _get_bool(payload: dict[str, object], key: str) -> bool | None:
        value = payload.get(key)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _get_float(payload: dict[str, object], key: str) -> float | None:
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    @classmethod
    def _normalize_parameters(cls, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            cleaned = cls._prune_nulls(item)
            if cleaned is not None:
                normalized[key] = cleaned
        return normalized

    @classmethod
    def _prune_nulls(cls, value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, dict):
            cleaned = {str(key): item for key, item in value.items() if item is not None}
            return cleaned or None
        if isinstance(value, list):
            cleaned = [item for item in value if item is not None]
            return cleaned or None
        return value

    @classmethod
    def _execution_plan(cls, value: object) -> list[ExecutionPlanStep]:
        if not isinstance(value, list):
            return []
        steps: list[ExecutionPlanStep] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            step = dict(item)
            step['parameters'] = cls._normalize_parameters(step.get('parameters'))
            step['dependsOn'] = cls._string_list(step.get('dependsOn') or step.get('depends_on'))
            bindings = step.get('parameterBindings') or step.get('parameter_bindings') or step.get('bindings')
            step['parameterBindings'] = bindings if isinstance(bindings, dict) else {}
            if 'step_id' in step and 'stepId' not in step:
                step['stepId'] = step['step_id']
            if 'visible_in_response' in step and 'visibleInResponse' not in step:
                step['visibleInResponse'] = step['visible_in_response']
            steps.append(ExecutionPlanStep.model_validate(step))
        return steps

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None and str(item)]
        return []

    @staticmethod
    def _derive_service(adapter: str, entity: str | None, operation: str | None) -> str | None:
        if entity and operation:
            normalized = adapter.strip()
            return normalized or None
        return None

    @staticmethod
    def _compose_intent(service: str | None, entity: str | None, operation: str | None) -> str | None:
        parts = [part for part in (service, entity, operation) if part]
        return '.'.join(parts) if len(parts) >= 3 else None

