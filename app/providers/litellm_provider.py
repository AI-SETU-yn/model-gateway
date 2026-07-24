import asyncio
import logging

import litellm

from app.config.settings import Settings
from app.exceptions.errors import GatewayTimeoutError, InvalidProviderResponseError, ProviderUnavailableError
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse, UsageResponse
from app.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseProvider):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        litellm.drop_params = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        last_error: Exception | None = None

        for attempt in range(self._settings.max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=f"ollama/{self._settings.model_name}",
                    api_base=self._settings.ollama_base_url,
                    messages=[
                        {'role': 'system', 'content': request.system_prompt},
                        {'role': 'user', 'content': request.user_prompt},
                    ],
                    timeout=self._settings.request_timeout_seconds,
                    metadata=request.metadata,
                )
                return self._to_response(response)
            except TimeoutError as exc:
                last_error = exc
                logger.warning('litellm_timeout', extra={'attempt': attempt + 1, 'model_name': self._settings.model_name})
            except Exception as exc:
                last_error = exc
                logger.warning('litellm_failure', extra={'attempt': attempt + 1, 'model_name': self._settings.model_name, 'error': str(exc)})

            if attempt < self._settings.max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))

        if isinstance(last_error, TimeoutError):
            raise GatewayTimeoutError('LiteLLM request timed out.') from last_error
        raise ProviderUnavailableError('LiteLLM or Ollama is unavailable.') from last_error

    def _to_response(self, body) -> GenerateResponse:
        try:
            message = body.choices[0].message
            content = getattr(message, 'content', None)
            if content is None and isinstance(message, dict):
                content = message.get('content')
            usage = getattr(body, 'usage', None)
            model = str(getattr(body, 'model', None) or f"ollama/{self._settings.model_name}")
            prompt_tokens = int(getattr(usage, 'prompt_tokens', 0) if usage is not None else 0)
            completion_tokens = int(getattr(usage, 'completion_tokens', 0) if usage is not None else 0)
            total_tokens = int(getattr(usage, 'total_tokens', 0) if usage is not None else 0)
            return GenerateResponse(
                response=str(content or ''),
                usage=UsageResponse(
                    promptTokens=prompt_tokens,
                    completionTokens=completion_tokens,
                    totalTokens=total_tokens,
                ),
                model=model,
            )
        except Exception as exc:
            raise InvalidProviderResponseError('Invalid response from LiteLLM provider.') from exc
