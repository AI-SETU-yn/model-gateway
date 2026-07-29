"""Service dependency wiring for the LiteLLM-backed gateway."""

from __future__ import annotations

from functools import lru_cache

from app.clients.litellm_client import LiteLLMClient
from app.config.settings import get_model_config, get_settings
from app.observability.inference import InferenceObserver
from app.providers.base import BaseInferenceProvider
from app.providers.litellm_provider import LiteLLMProvider
from app.registry.model_registry import ModelRegistry
from app.registry.prompt_registry import PromptRegistry
from app.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.retry.policy import RetryExecutor, RetryPolicy
from app.services.generation.service import ResponseGenerationService
from app.services.general_chat.service import GeneralChatService
from app.services.health_service import HealthService
from app.services.inference import ERPInferenceService, GeneralInferenceService
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
def get_litellm_client() -> LiteLLMClient:
    return LiteLLMClient(get_settings())


@lru_cache(maxsize=1)
def get_retry_policy() -> RetryPolicy:
    settings = get_settings()
    return RetryPolicy(max_retries=settings.litellm_max_retries, timeout_seconds=settings.litellm_timeout_seconds)


@lru_cache(maxsize=1)
def get_retry_executor() -> RetryExecutor:
    return RetryExecutor(get_retry_policy())


@lru_cache(maxsize=1)
def get_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(CircuitBreakerConfig())


@lru_cache(maxsize=1)
def get_inference_observer() -> InferenceObserver:
    return InferenceObserver()


@lru_cache(maxsize=1)
def get_inference_provider() -> BaseInferenceProvider:
    return LiteLLMProvider(get_litellm_client(), get_retry_executor(), get_inference_observer(), get_circuit_breaker())


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
