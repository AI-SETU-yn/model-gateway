from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InferenceMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str


class InferenceUsage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt_tokens: int = Field(default=0, alias='promptTokens')
    completion_tokens: int = Field(default=0, alias='completionTokens')
    total_tokens: int = Field(default=0, alias='totalTokens')


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    model_alias: str = Field(alias='modelAlias')
    system_prompt: str = Field(alias='systemPrompt')
    user_prompt: str = Field(alias='userPrompt')
    conversation_history: list[InferenceMessage] = Field(default_factory=list, alias='conversationHistory')
    adapter: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, alias='maxTokens')
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.extend(message.model_dump() for message in self.conversation_history)
        if self.user_prompt:
            messages.append({'role': 'user', 'content': self.user_prompt})
        return messages


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    content: str
    finish_reason: str | None = Field(default=None, alias='finishReason')
    usage: InferenceUsage = Field(default_factory=InferenceUsage)
    latency_ms: float = Field(alias='latencyMs')
    provider: str
    model: str
    adapter: str | None = None
    request_id: str | None = Field(default=None, alias='requestId')
    trace_id: str | None = Field(default=None, alias='traceId')
    stream: bool = False
