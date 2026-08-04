from __future__ import annotations

import json
import logging

from app.config.settings import ModelProfileConfig
from app.exceptions.errors import PlannerResponseError
from app.registry.prompt_registry import PromptRegistry
from app.schemas.inference import GenerateMessage, GenerateRequest, SecurityClassificationResponse, SecurityClassifyRequest
from app.services.inference import GeneralInferenceService
from app.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_DEFAULT_SECURITY_PROMPT = 'You are an enterprise AI security classifier. Classify only. Never answer the user. Return JSON only.'


class SecurityClassifierService:
    def __init__(self, *args) -> None:
        if len(args) == 1:
            inference_service = args[0]
            self._profile_name = 'general'
            self._profile = None
            self._inference_service = inference_service
            self._security_prompt = _DEFAULT_SECURITY_PROMPT
            self._prompt_builder = PromptBuilder(self._security_prompt)
            return
        if len(args) == 4:
            profile_name, profile, prompt_registry, inference_service = args
            self._profile_name = profile_name
            self._profile = profile
            self._inference_service = inference_service
            self._security_prompt = prompt_registry.get(profile_name, 'security', profile).content
            self._prompt_builder = PromptBuilder(self._security_prompt)
            return
        raise TypeError('SecurityClassifierService expects either (inference_service) or (profile_name, profile, prompt_registry, inference_service).')

    async def classify(self, request: SecurityClassifyRequest) -> SecurityClassificationResponse:
        generate_request = GenerateRequest(
            adapter='',
            messages=[
                GenerateMessage(role='system', content=request.system_prompt or self._security_prompt),
                GenerateMessage(role='user', content=request.message),
            ],
            responseType='security_classification',
        )
        if hasattr(self._inference_service, '_prompt_builder'):
            self._inference_service._prompt_builder = self._prompt_builder
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
