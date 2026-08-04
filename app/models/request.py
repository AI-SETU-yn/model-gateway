"""Request and health request models for the Yn AI Setu Model Gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """Single chat message in OpenAI-compatible format."""

    model_config = ConfigDict(extra='forbid')

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str

    @field_validator('content')
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError('message content must not be empty.')
        return normalized


class ChatRequest(BaseModel):
    """Inbound chat completion request."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = Field(default=1024, alias='max_tokens')
    stream: bool = False

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError('model must not be blank when provided.')
        return normalized

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError('temperature must be between 0 and 2.')
        return value

    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError('max_tokens must be greater than zero.')
        return value

    @field_validator('stream')
    @classmethod
    def validate_stream(cls, value: bool) -> bool:
        return bool(value)


class HealthCheckRequest(BaseModel):
    """Optional structured health probe request."""

    model_config = ConfigDict(extra='forbid')

    model: str | None = None


GenerateRequest = ChatRequest

from app.models.response import HealthCheckResponse as HealthResponse  # noqa: E402
