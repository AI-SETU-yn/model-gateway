"""Service dependency wiring for the self-hosted Transformers gateway."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config.settings import get_model_config
from app.providers.base import BaseInferenceProvider
from app.providers.transformers_provider import TransformersProvider
from app.registry.model_registry import ModelRegistry
from app.registry.prompt_registry import PromptRegistry
from app.services.adapter_manager import AdapterManager
from app.services.generation.service import ResponseGenerationService
from app.services.general_chat.service import GeneralChatService
from app.services.health_service import HealthService
from app.services.inference import ERPInferenceService, GeneralInferenceService
from app.services.loading.general_model_loader import GeneralModelLoader
from app.services.loading.erp_model_loader import ERPModelLoader
from app.services.metrics_service import MetricsService
from app.services.planner import PlannerService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter
from app.services.security_classifier import SecurityClassifierService


@lru_cache(maxsize=1)
def get_metrics_service() -> MetricsService:
    return MetricsService()


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(get_model_config())


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()


@lru_cache(maxsize=1)
def get_adapter_manager(adapters_root: str) -> AdapterManager:
    project_root = Path(__file__).resolve().parents[2]
    return AdapterManager(project_root, Path(adapters_root))


@lru_cache(maxsize=1)
def get_erp_loader() -> ERPModelLoader:
    registry = get_model_registry()
    profile_name, profile = registry.generation_profile()
    return ERPModelLoader(profile_name, profile, get_adapter_manager(profile.adapters_root))


@lru_cache(maxsize=1)
def get_general_loader() -> GeneralModelLoader:
    registry = get_model_registry()
    profile_name, profile = registry.general_chat_profile()
    return GeneralModelLoader(profile_name, profile)


@lru_cache(maxsize=1)
def get_inference_provider() -> BaseInferenceProvider:
    return TransformersProvider({'erp': get_erp_loader(), 'general': get_general_loader()})


@lru_cache(maxsize=1)
def get_health_service() -> HealthService:
    return HealthService(get_inference_provider())


@lru_cache(maxsize=1)
def get_erp_inference_service() -> ERPInferenceService:
    registry = get_model_registry()
    prompts = get_prompt_registry()
    profile_name, profile = registry.generation_profile()
    prompt_builder = PromptBuilder(prompts.get(profile_name, 'generator', profile).content)
    return ERPInferenceService(profile_name, profile, get_inference_provider(), get_metrics_service(), prompt_builder, ResponseFormatter())


@lru_cache(maxsize=1)
def get_general_inference_service() -> GeneralInferenceService:
    registry = get_model_registry()
    prompts = get_prompt_registry()
    profile_name, profile = registry.general_chat_profile()
    prompt_builder = PromptBuilder(prompts.get(profile_name, 'chat', profile).content)
    return GeneralInferenceService(profile_name, profile, get_inference_provider(), get_metrics_service(), prompt_builder, ResponseFormatter())


@lru_cache(maxsize=1)
def get_generation_service() -> ResponseGenerationService:
    return ResponseGenerationService(get_erp_inference_service())


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    registry = get_model_registry()
    prompts = get_prompt_registry()
    profile_name, profile = registry.planner_profile()
    return PlannerService(profile_name, profile, prompts, get_erp_inference_service())


@lru_cache(maxsize=1)
def get_security_classifier_service() -> SecurityClassifierService:
    registry = get_model_registry()
    prompts = get_prompt_registry()
    profile_name, profile = registry.security_profile()
    return SecurityClassifierService(profile_name, profile, prompts, get_general_inference_service())


@lru_cache(maxsize=1)
def get_general_chat_service() -> GeneralChatService:
    registry = get_model_registry()
    prompts = get_prompt_registry()
    profile_name, profile = registry.general_chat_profile()
    return GeneralChatService(profile_name, profile, prompts, get_general_inference_service())
