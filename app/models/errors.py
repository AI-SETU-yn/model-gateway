from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    code: str
    message: str
    request_id: str | None = None
    correlation_id: str | None = None
