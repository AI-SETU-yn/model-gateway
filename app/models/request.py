from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    system_prompt: str = Field(alias='systemPrompt')
    user_prompt: str = Field(alias='userPrompt')
    conversation_id: str | None = Field(default=None, alias='conversationId')
    metadata: dict[str, object] = Field(default_factory=dict)


class UsageResponse(BaseModel):
    prompt_tokens: int = Field(alias='promptTokens')
    completion_tokens: int = Field(alias='completionTokens')
    total_tokens: int = Field(alias='totalTokens')


class GenerateResponse(BaseModel):
    response: str
    usage: UsageResponse
    model: str


class HealthResponse(BaseModel):
    status: str
