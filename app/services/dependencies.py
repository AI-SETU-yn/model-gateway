"""Service dependency wiring for the self-hosted Transformers gateway."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config.settings import get_model_config, get_settings
from app.providers.base import BaseInferenceProvider
from app.providers.configured import ConfiguredInferenceProvider
from app.registry.model_registry import ModelRegistry
from app.registry.prompt_registry import PromptRegistry
from app.services.adapter_manager import AdapterManager
from app.services.generation.service import ResponseGenerationService
from app.services.general_chat.service import GeneralChatService
from app.services.health_service import HealthService
from app.services.inference import ERPInferenceService, GeneralInferenceService
from app.services.loading.base import BaseModelLoader
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


@lru_cache(maxsize=32)
def get_transformers_loader(profile_name: str) -> BaseModelLoader:
    from app.services.loading.erp_model_loader import ERPModelLoader
    from app.services.loading.general_model_loader import GeneralModelLoader

    registry = get_model_registry()
    profile = registry.get(profile_name)
    if profile.adapter_enabled:
        return ERPModelLoader(profile_name, profile, get_adapter_manager(profile.adapters_root))
    return GeneralModelLoader(profile_name, profile)


@lru_cache(maxsize=1)
def get_erp_loader() -> BaseModelLoader:
    registry = get_model_registry()
    profile_name, _ = registry.generation_profile()
    return get_transformers_loader(profile_name)


def get_erp_model_loader():
    return get_erp_loader()


@lru_cache(maxsize=1)
def get_general_loader() -> BaseModelLoader:
    registry = get_model_registry()
    profile_name, _ = registry.general_chat_profile()
    return get_transformers_loader(profile_name)


def get_general_model_loader():
    return get_general_loader()


@lru_cache(maxsize=1)
def get_transformers_provider() -> BaseInferenceProvider:
    from app.providers.transformers_provider import TransformersProvider

    registry = get_model_registry()
    loaders = {
        profile_name: get_transformers_loader(profile_name)
        for profile_name, profile in registry.all_profiles().items()
        if profile.provider.strip().lower() == 'transformers'
    }
    return TransformersProvider(loaders)


@lru_cache(maxsize=1)
def get_litellm_provider() -> BaseInferenceProvider:
    from app.clients.litellm_client import LiteLLMClient
    from app.observability.inference import InferenceObserver
    from app.providers.litellm_provider import LiteLLMProvider
    from app.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from app.retry.policy import RetryExecutor, RetryPolicy

    settings = get_settings()
    return LiteLLMProvider(
        LiteLLMClient(settings, get_model_config()),
        RetryExecutor(RetryPolicy(max_retries=settings.litellm_max_retries, timeout_seconds=settings.litellm_timeout_seconds)),
        InferenceObserver(),
        CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=settings.litellm_circuit_failure_threshold,
                recovery_timeout_seconds=settings.litellm_circuit_recovery_timeout_seconds,
            )
        ),
    )


@lru_cache(maxsize=1)
def get_inference_provider() -> BaseInferenceProvider:
    return ConfiguredInferenceProvider(get_model_config(), get_transformers_provider, get_litellm_provider)


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
    return ResponseGenerationService(get_erp_inference_service(), get_general_inference_service())


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
