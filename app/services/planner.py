"""Planner inference service."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config.settings import ModelProfileConfig
from app.exceptions.errors import PlannerResponseError
from app.registry.prompt_registry import PromptRegistry
from app.schemas.inference import PlannerRequest, PlannerResponse
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
        prompt = self._build_prompt(request.query)
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
            model=model_name,
        )

    def _fallback_plan(self, request: PlannerRequest, *, raw_response: str, model_name: str) -> PlannerResponse:
        target = self._select_fallback_target(request.query, request.adapter)
        if target is None:
            return PlannerResponse(
                intent='general.chat', tool=None, domain=None, service=None, entity=None, operation=None,
                parameters={}, requiresTool=False, responseType='text', confidence=0.0,
                rawResponse=raw_response, adapter=request.adapter, model=model_name,
            )
        service = target.service or request.adapter
        intent = target.intent or self._compose_intent(service, target.entity, target.operation)
        return PlannerResponse(
            intent=intent, tool=target.tool, domain=target.domain, service=service, entity=target.entity,
            operation=target.operation, parameters={}, requiresTool=True, responseType=target.response_type,
            confidence=0.0, rawResponse=raw_response, adapter=request.adapter, model=model_name,
        )

    def _select_fallback_target(self, query: str, adapter: str):
        query_tokens = self._tokens(query)
        if not query_tokens:
            return None
        scored = []
        for target in self._profile.planner_targets:
            if target.service and target.service != adapter:
                continue
            score = self._score_fallback_target(target, query, query_tokens)
            if score > 0:
                scored.append((score, target))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        best = [target for score, target in scored if score == best_score]
        return best[0] if len(best) == 1 else None

    @classmethod
    def _score_fallback_target(cls, target, query: str, query_tokens: list[str]) -> float:
        query_token_set = set(query_tokens)
        normalized_query = ' '.join(query_tokens)
        entity_tokens = cls._tokens(target.entity)
        target_tokens = cls._tokens(' '.join(item for item in (target.intent, target.tool, target.entity, target.operation, target.description, ' '.join(target.keywords)) if item))
        score = 0.0
        if entity_tokens:
            overlap = len(set(entity_tokens) & query_token_set)
            score += (overlap / len(set(entity_tokens))) * 50
            positions = [query_tokens.index(token) for token in entity_tokens if token in query_token_set]
            if positions:
                score += 30 / (min(positions) + 1)
        score += len(set(target_tokens) & query_token_set)
        for keyword in target.keywords:
            keyword_tokens = cls._tokens(keyword)
            if keyword_tokens and ' '.join(keyword_tokens) in normalized_query:
                score += 15
        return score

    def _build_prompt(self, query: str) -> str:
        planner_prompt = self._prompt_registry.planner_prompt(self._profile)
        return (
            f"{planner_prompt}\n"
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

    @staticmethod
    def _tokens(value: str) -> list[str]:
        normalized = re.sub(r'[^a-zA-Z0-9]+', ' ', value).lower()
        tokens: list[str] = []
        for token in normalized.split():
            if token in {'a', 'an', 'and', 'are', 'for', 'give', 'me', 'my', 'of', 'show', 'the', 'to'}:
                continue
            tokens.append(token[:-1] if token.endswith('s') and len(token) > 3 else token)
        return tokens
