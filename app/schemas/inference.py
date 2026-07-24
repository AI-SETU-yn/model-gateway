from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: str
    prompt: str


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


class ReloadAdapterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: str


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    intent: str | None = None
    tool: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    raw_response: str = Field(alias='rawResponse')
    adapter: str
    model: str
