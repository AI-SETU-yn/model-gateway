from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerateMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str


class GenerationPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    grounded: bool = True
    hallucination: str = 'forbid'
    output_format: str = Field(default='markdown', alias='format')
    use_tool_results_only: bool = Field(default=True, alias='useToolResultsOnly')
    never_invent_business_data: bool = Field(default=True, alias='neverInventBusinessData')
    never_ask_for_present_data: bool = Field(default=True, alias='neverAskForPresentData')
    ignore_orchestration_metadata: bool = Field(default=True, alias='ignoreOrchestrationMetadata')
    concise: bool = True


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    user_question: str | None = Field(default=None, alias='userQuestion')
    planner_intent: str | None = Field(default=None, alias='plannerIntent')
    execution_plan: list[dict[str, Any]] = Field(default_factory=list, alias='executionPlan')


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    adapter: str = ''
    messages: list[GenerateMessage] | None = None
    tool_result: Any | None = Field(default=None, alias='toolResult')
    response_type: str | None = Field(default=None, alias='responseType')
    generation_policy: GenerationPolicy | None = Field(default=None, alias='generationPolicy')
    conversation: ConversationContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    missing_parameters: list[str] = Field(default_factory=list, alias='missingParameters')

    @model_validator(mode='after')
    def require_generation_input(self) -> 'GenerateRequest':
        if not self.messages and self.tool_result is None:
            raise ValueError('GenerateRequest requires messages or toolResult.')
        return self


class GeneralChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    prompt: str | None = None
    messages: list[GenerateMessage] | None = None
    response_type: str | None = Field(default='general_chat', alias='responseType')

    @model_validator(mode='after')
    def require_chat_input(self) -> 'GeneralChatRequest':
        if self.prompt is None and not self.messages:
            raise ValueError('GeneralChatRequest requires prompt or messages.')
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
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    adapter: str = ''
    query: str
    prompt: str | None = None
    registry_context: str | None = Field(default=None, alias='registryContext')
    registry_metadata: Any | None = Field(default=None, alias='registryMetadata')


class ExecutionPlanStep(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    step_id: str | None = Field(default=None, alias='stepId')
    intent: str | None = None
    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    tool: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, alias='dependsOn')
    parameter_bindings: dict[str, Any] = Field(default_factory=dict, alias='parameterBindings')
    question: str | None = None
    visible_in_response: bool | None = Field(default=None, alias='visibleInResponse')


class SecurityClassifyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    message: str
    system_prompt: str = Field(alias='system_prompt')


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
    execution_plan: list[ExecutionPlanStep] = Field(default_factory=list, alias='executionPlan')
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


class ModelHealthItem(BaseModel):
    model_config = ConfigDict(extra='forbid')

    profile: str
    model: str
    provider: str
    base_url: str = Field(alias='baseUrl')
    status: str
    latency_ms: float = Field(alias='latencyMs')
    adapter_enabled: bool = Field(alias='adapterEnabled')
    adapter: str | None = None
    last_checked: str = Field(alias='lastChecked')


class ModelHealthResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    models: list[ModelHealthItem]
