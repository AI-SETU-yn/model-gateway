"""Planner inference service."""

from __future__ import annotations

import json
import logging

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import PlannerResponseError
from app.schemas.inference import PlannerRequest, PlannerResponse
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)


class PlannerService:
    """Generate planner JSON using the inference model."""

    def __init__(self, config: ModelGatewayConfig, inference_service: InferenceService) -> None:
        self._config = config
        self._inference_service = inference_service

    async def plan(self, request: PlannerRequest) -> PlannerResponse:
        prompt = self._build_prompt(request.query)
        raw_response, _ = await self._inference_service.planner_generate(request.adapter, prompt)
        parsed = self._parse_json(raw_response)
        return PlannerResponse(
            intent=parsed.get('intent'),
            tool=parsed.get('tool'),
            parameters=parsed.get('parameters', {}),
            rawResponse=raw_response,
            adapter=request.adapter,
            model=self._config.base_model,
        )

    def _build_prompt(self, query: str) -> str:
        return (
            f"{self._config.planner_system_prompt}\n"
            'Schema: {"intent": string|null, "tool": string|null, "parameters": object}.\n'
            f'User Query: {query}\n'
            'JSON:'
        )

    @staticmethod
    def _parse_json(raw_response: str) -> dict[str, object]:
        candidate = raw_response.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find('{')
            end = candidate.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError as exc:
                    raise PlannerResponseError('Planner model did not return valid JSON.') from exc
            raise PlannerResponseError('Planner model did not return valid JSON.')
