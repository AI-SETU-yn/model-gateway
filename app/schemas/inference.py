from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerateMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str


class GenerationPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    use_tool_results_only: bool = Field(default=True, alias='useToolResultsOnly')
    never_invent_business_data: bool = Field(default=True, alias='neverInventBusinessData')
    never_ask_for_present_data: bool = Field(default=True, alias='neverAskForPresentData')
    ignore_orchestration_metadata: bool = Field(default=True, alias='ignoreOrchestrationMetadata')
    concise: bool = True


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    adapter: str
    prompt: str | None = None
    messages: list[GenerateMessage] | None = None
    tool_result: Any | None = Field(default=None, alias='toolResult')
    response_type: str | None = Field(default=None, alias='responseType')
    generation_policy: GenerationPolicy | None = Field(default=None, alias='generationPolicy')

    @model_validator(mode='after')
    def require_generation_input(self) -> 'GenerateRequest':
        if self.prompt is None and not self.messages and self.tool_result is None:
            raise ValueError('GenerateRequest requires prompt, messages, or toolResult.')
        return self


class UsageResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt_tokens: int = Field(alias='promptTokens')
    completion_tokens: int = Field(alias='completionTokens')
    total_tokens: int = Field(alias='totalTokens')


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    response: str
    adapter: str
    model: str
    usage: UsageResponse
    generation_time_ms: float = Field(alias='generationTimeMs')


class PlannerRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: str
    query: str


class SecurityClassifyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    message: str
    system_prompt: str = Field(alias='system_prompt')
    adapter: str | None = None


class ReloadAdapterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: str


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    intent: str | None = None
    tool: str | None = None
    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    requires_tool: bool | None = Field(default=None, alias='requiresTool')
    response_type: str | None = Field(default=None, alias='responseType')
    confidence: float | None = None
    raw_response: str = Field(alias='rawResponse')
    adapter: str
    model: str


class SecurityClassificationResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    safe: bool
    category: str
    confidence: float
    reason: str