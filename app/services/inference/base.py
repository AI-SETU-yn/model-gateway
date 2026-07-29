"""LiteLLM-backed inference services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.settings import ModelProfileConfig
from app.models.inference import InferenceMessage, InferenceRequest, InferenceResponse
from app.providers.base import BaseInferenceProvider
from app.schemas.inference import GenerateRequest, GenerateResponse
from app.services.metrics_service import MetricsService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    finish_reason: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    adapter: str | None = None


class BaseInferenceService:
    def __init__(
        self,
        profile_name: str,
        profile: ModelProfileConfig,
        provider: BaseInferenceProvider,
        metrics_service: MetricsService,
        prompt_builder: PromptBuilder,
        response_formatter: ResponseFormatter,
    ) -> None:
        self._profile_name = profile_name
        self._profile = profile
        self._provider = provider
        self._metrics_service = metrics_service
        self._prompt_builder = prompt_builder
        self._response_formatter = response_formatter

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def model_name(self) -> str:
        return self._profile.model_name

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        bundle = self._prompt_builder.build(request)
        result = await self._complete(
            messages=bundle.messages,
            adapter=request.adapter or None,
            max_tokens=self._profile.generation.max_new_tokens,
            temperature=self._profile.generation.temperature,
            top_p=self._profile.generation.top_p,
            metadata={'profile': self._profile_name, 'response_type': request.response_type or ''},
        )
        self._metrics_service.record_generation(result.elapsed_ms)
        return self._response_formatter.format_generate_response(
            text=result.text,
            adapter=request.adapter or '',
            model=result.model or self._profile.model_name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            elapsed_ms=result.elapsed_ms,
        )

    async def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        adapter: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        metadata: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> CompletionResult:
        return await self._complete(
            messages=messages,
            adapter=adapter,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
            stream=stream,
        )

    async def _complete(
        self,
        *,
        messages: list[dict[str, str]],
        adapter: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        metadata: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> CompletionResult:
        history = [InferenceMessage(role=message['role'], content=message['content']) for message in messages[1:-1]] if len(messages) > 2 else []
        system_prompt = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ''
        user_prompt = messages[-1]['content'] if messages else ''
        provider_request = InferenceRequest(
            modelAlias=self._profile_name,
            systemPrompt=system_prompt,
            userPrompt=user_prompt,
            conversationHistory=history,
            adapter=adapter,
            temperature=temperature,
            top_p=top_p,
            maxTokens=max_tokens,
            stream=stream,
            metadata={
                **(metadata or {}),
                'provider': self._profile.provider,
                'model_name': self._profile.model_name,
                'api_base': self._profile.base_url,
                'api_key': self._profile.api_key,
            },
        )
        response: InferenceResponse = await self._provider.complete(provider_request)
        return CompletionResult(
            text=response.content,
            model=response.model or self._profile.model_name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            elapsed_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            request_id=response.request_id,
            trace_id=response.trace_id,
            adapter=response.adapter,
        )
