from __future__ import annotations

import json
import logging

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import InvalidRequestError, PlannerResponseError
from app.schemas.inference import GenerateMessage, GenerateRequest, SecurityClassificationResponse, SecurityClassifyRequest
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)


class SecurityClassifierService:
    def __init__(self, config: ModelGatewayConfig, inference_service: InferenceService) -> None:
        self._config = config
        self._inference_service = inference_service

    async def classify(self, request: SecurityClassifyRequest) -> SecurityClassificationResponse:
        adapter = (request.adapter or self._config.default_adapter or '').strip()
        if not adapter:
            raise InvalidRequestError('Security classifier adapter is not configured.')

        generate_request = GenerateRequest(
            adapter=adapter,
            messages=[
                GenerateMessage(role='system', content=request.system_prompt),
                GenerateMessage(role='user', content=request.message),
            ],
            responseType='security_classification',
        )
        response = await self._inference_service.generate(generate_request)
        payload = self._parse_json(response.response)
        try:
            return SecurityClassificationResponse.model_validate(payload)
        except Exception as exc:
            raise PlannerResponseError('Security classifier did not return a valid JSON object.') from exc

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
                    raise PlannerResponseError('Security classifier did not return valid JSON.') from exc
            else:
                raise PlannerResponseError('Security classifier did not return valid JSON.')
        if not isinstance(parsed, dict):
            raise PlannerResponseError('Security classifier did not return a JSON object.')
        logger.info('security_classifier_model_response_json=%s', parsed)
        return parsed