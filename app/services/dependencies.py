"""Service dependency wiring."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config.settings import get_model_config, get_settings
from app.services.adapter_manager import AdapterManager
from app.services.chat_template import ChatTemplateHandler
from app.services.inference import InferenceService
from app.services.metrics_service import MetricsService
from app.services.model_loader import ModelLoader
from app.services.planner import PlannerService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter


@lru_cache(maxsize=1)
def get_adapter_manager() -> AdapterManager:
    config = get_model_config()
    project_root = Path(__file__).resolve().parents[2]
    return AdapterManager(project_root, config.adapters_root)


@lru_cache(maxsize=1)
def get_metrics_service() -> MetricsService:
    return MetricsService()


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    return ModelLoader(get_model_config(), get_adapter_manager())


@lru_cache(maxsize=1)
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder(get_model_config().generate_system_prompt)


@lru_cache(maxsize=1)
def get_chat_template_handler() -> ChatTemplateHandler:
    return ChatTemplateHandler()


@lru_cache(maxsize=1)
def get_response_formatter() -> ResponseFormatter:
    return ResponseFormatter()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    settings = get_settings()
    return InferenceService(
        get_model_config(),
        get_model_loader(),
        get_metrics_service(),
        settings.max_concurrent_requests,
        get_prompt_builder(),
        get_chat_template_handler(),
        get_response_formatter(),
    )


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    return PlannerService(get_model_config(), get_inference_service())
