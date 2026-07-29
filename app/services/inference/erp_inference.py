from __future__ import annotations

from app.clients.base import BaseInferenceClient
from app.config.settings import ModelProfileConfig
from app.services.inference.base import BaseInferenceService, CompletionResult
from app.services.metrics_service import MetricsService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter


class ERPInferenceService(BaseInferenceService):
    def __init__(
        self,
        profile_name: str,
        profile: ModelProfileConfig,
        client: BaseInferenceClient,
        metrics_service: MetricsService,
        prompt_builder: PromptBuilder,
        response_formatter: ResponseFormatter,
    ) -> None:
        super().__init__(profile_name, profile, client, metrics_service, prompt_builder, response_formatter)

    async def planner_generate(self, adapter: str, prompt: str) -> tuple[str, float, str]:
        result: CompletionResult = await self.complete_messages(
            messages=[{'role': 'user', 'content': prompt}],
            adapter=adapter,
            max_tokens=self._profile.generation.planner_max_new_tokens,
            temperature=self._profile.generation.temperature,
            top_p=self._profile.generation.top_p,
            metadata={'profile': self._profile_name, 'mode': 'planner'},
        )
        self._metrics_service.record_planner(result.elapsed_ms)
        return result.text, result.elapsed_ms, result.model
