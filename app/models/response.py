"""Response models for the Yn AI Setu Model Gateway."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AssistantMessage(BaseModel):
    """Assistant message returned in a choice."""

    model_config = ConfigDict(extra='allow')

    role: Literal['assistant'] = 'assistant'
    content: str


class ChatChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    index: int
    message: AssistantMessage
    finish_reason: str | None = Field(default=None, alias='finish_reason')


class Usage(BaseModel):
    """Token usage summary."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    prompt_tokens: int = Field(default=0, alias='prompt_tokens')
    completion_tokens: int = Field(default=0, alias='completion_tokens')
    total_tokens: int = Field(default=0, alias='total_tokens')


class ChatCompletionResponse(BaseModel):
    """Standardized chat completion response."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    id: str
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelCard(BaseModel):
    """Single remotely available model entry."""

    model_config = ConfigDict(extra='allow')

    id: str
    object: str = 'model'
    owned_by: str = 'gateway'


class ModelListResponse(BaseModel):
    """OpenAI-compatible model listing response."""

    model_config = ConfigDict(extra='allow')

    object: str = 'list'
    data: list[ModelCard]


class HealthCheckResponse(BaseModel):
    """Gateway health response."""

    model_config = ConfigDict(extra='forbid')

    status: str
    service: str
    checks: dict[str, Any]
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class ErrorResponse(BaseModel):
    """Error payload returned by exception handlers."""

    model_config = ConfigDict(extra='forbid')

    code: str
    message: str
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


GenerateResponse = ChatCompletionResponse
HealthResponse = HealthCheckResponse
UsageResponse = Usage

__all__ = [
    'AssistantMessage',
    'ChatChoice',
    'ChatCompletionResponse',
    'Usage',
    'ModelCard',
    'ModelListResponse',
    'HealthCheckResponse',
    'ErrorResponse',
    'GenerateResponse',
    'HealthResponse',
    'UsageResponse',
]
