"""Planner inference service."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import PlannerResponseError
from app.schemas.inference import PlannerRequest, PlannerResponse

if TYPE_CHECKING:
    from app.services.inference import InferenceService

logger = logging.getLogger(__name__)


class PlannerService:
    """Generate planner JSON using the inference model."""

    def __init__(self, config: ModelGatewayConfig, inference_service: 'InferenceService') -> None:
        self._config = config
        self._inference_service = inference_service

    async def plan(self, request: PlannerRequest) -> PlannerResponse:
        prompt = self._build_prompt(request.query)
        raw_response, _ = await self._inference_service.planner_generate(request.adapter, prompt)
        parsed = self._parse_json(raw_response)

        domain = self._get_text(parsed, 'domain')
        entity = self._get_text(parsed, 'entity')
        operation = self._get_text(parsed, 'operation')
        service = self._get_text(parsed, 'service') or self._derive_service(request.adapter, entity, operation)
        intent = self._get_text(parsed, 'intent') or self._compose_intent(service, entity, operation)
        tool = self._get_text(parsed, 'tool')
        parameters = self._normalize_parameters(parsed.get('parameters'))
        requires_tool = self._get_bool(parsed, 'requiresTool')
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
            confidence=confidence,
            rawResponse=raw_response,
            adapter=request.adapter,
            model=self._config.base_model,
        )

    def _build_prompt(self, query: str) -> str:
        return (
            f"{self._config.planner_system_prompt}\n"
            'Schema: {"domain": string|null, "service": string|null, "entity": string|null, '
            '"operation": string|null, "intent": string|null, "tool": string|null, '
            '"parameters": object, "requiresTool": boolean|null, "responseType": string|null, '
            '"confidence": number|null}.\n'
            'Return only valid JSON. If you do not know an exact tool name, set "tool" to null.\n'
            f'User Query: {query}\n'
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
        if isinstance(value, bool):
            return value
        return None

    @staticmethod
    def _get_float(payload: dict[str, object], key: str) -> float | None:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

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
            cleaned_dict: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                cleaned_item = cls._prune_nulls(item)
                if cleaned_item is not None:
                    cleaned_dict[key] = cleaned_item
            return cleaned_dict or None
        if isinstance(value, list):
            cleaned_list = [item for item in (cls._prune_nulls(item) for item in value) if item is not None]
            return cleaned_list or None
        return value

    @staticmethod
    def _derive_service(adapter: str, entity: str | None, operation: str | None) -> str | None:
        if entity and operation:
            normalized = adapter.strip()
            return normalized or None
        return None

    @staticmethod
    def _compose_intent(service: str | None, entity: str | None, operation: str | None) -> str | None:
        parts = [part for part in (service, entity, operation) if part]
        if len(parts) < 3:
            return None
        return '.'.join(parts)
