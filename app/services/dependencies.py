"""Service dependency wiring."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Request

from app.config.settings import get_model_config, get_settings
from app.services.adapter_manager import AdapterManager
from app.services.inference import InferenceService
from app.services.metrics_service import MetricsService
from app.services.model_loader import ModelLoader
from app.services.planner import PlannerService


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
def get_inference_service() -> InferenceService:
    settings = get_settings()
    return InferenceService(
        get_model_config(),
        get_model_loader(),
        get_metrics_service(),
        settings.max_concurrent_requests,
    )


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    return PlannerService(get_model_config(), get_inference_service())
