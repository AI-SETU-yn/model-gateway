"""Response formatting for generation endpoints."""

from __future__ import annotations

from app.schemas.inference import GenerateResponse, UsageResponse


class ResponseFormatter:
    """Build public API responses from inference results."""

    def format_generate_response(
        self,
        *,
        text: str,
        adapter: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        elapsed_ms: float,
    ) -> GenerateResponse:
        return GenerateResponse(
            response=text,
            adapter=adapter,
            model=model,
            usage=UsageResponse(
                promptTokens=prompt_tokens,
                completionTokens=completion_tokens,
                totalTokens=total_tokens,
            ),
            generationTimeMs=round(elapsed_ms, 2),
        )
