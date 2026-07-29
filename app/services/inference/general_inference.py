from __future__ import annotations

from app.clients.base import BaseInferenceClient
from app.config.settings import ModelProfileConfig
from app.services.inference.base import BaseInferenceService
from app.services.metrics_service import MetricsService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter


class GeneralInferenceService(BaseInferenceService):
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
