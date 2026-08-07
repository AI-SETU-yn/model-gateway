"""Dependency providers for the Model Gateway.

Two independent provider chains are wired here, both ultimately backed by
LiteLLM:

- The generic OpenAI-compatible chat proxy (`get_settings`,
  `get_litellm_service`, `get_chat_service`) - re-exported from
  `app.core.config` - backs `/chat` and `/v1/models`.
- The per-capability inference subsystem (`get_generation_service`,
  `get_planner_service`, `get_security_classifier_service`,
  `get_general_chat_service`, `get_general_inference_service`,
  `get_model_registry`, `get_health_service`, `get_metrics_service`) backs
  `/planner`, `/generate`, `/security/classify`, `/chat/general`, `/health`,
  `/ready`, `/health/models`, and `/metrics` - the documented public API
  surface (see README.md). It routes per capability (planner/generate/
  security/general-chat can each use a different model profile, per
  `configs/model.yaml`'s `routing` section) through `LiteLLMProvider` for
  resilience (circuit breaker + retry) on top of `LiteLLMClient`.
"""

from __future__ import annotations

from functools import lru_cache

from app.clients.litellm_client import LiteLLMClient
from app.core.config import ModelGatewayConfig, get_chat_service, get_litellm_service, get_model_gateway_config, get_settings
from app.observability.inference import InferenceObserver
from app.providers.litellm_provider import LiteLLMProvider
from app.registry.model_registry import ModelRegistry
from app.registry.prompt_registry import PromptRegistry
from app.retry.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.retry.policy import RetryExecutor, RetryPolicy
from app.services.generation.service import ResponseGenerationService
from app.services.general_chat.service import GeneralChatService
from app.services.health_service import HealthService
from app.services.inference.erp_inference import ERPInferenceService
from app.services.inference.general_inference import GeneralInferenceService
from app.services.metrics_service import MetricsService
from app.services.planner import PlannerService
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter
from app.services.security_classifier import SecurityClassifierService

__all__ = [
    'get_settings',
    'get_litellm_service',
    'get_chat_service',
    'get_model_registry',
    'get_generation_service',
    'get_planner_service',
    'get_security_classifier_service',
    'get_general_chat_service',
    'get_general_inference_service',
    'get_health_service',
    'get_metrics_service',
]


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return the cached, validated per-capability model registry."""

    return ModelRegistry(get_model_gateway_config())


@lru_cache(maxsize=1)
def get_metrics_service() -> MetricsService:
    return MetricsService()


@lru_cache(maxsize=1)
def _get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()


@lru_cache(maxsize=1)
def _get_response_formatter() -> ResponseFormatter:
    return ResponseFormatter()


@lru_cache(maxsize=1)
def _get_litellm_client() -> LiteLLMClient:
    settings = get_settings()
    config: ModelGatewayConfig = get_model_gateway_config()
    return LiteLLMClient(settings, config)


@lru_cache(maxsize=1)
def _get_litellm_provider() -> LiteLLMProvider:
    """Return the cached, resilient (circuit-breaker + retry) LiteLLM provider.

    Shared across every profile: `LiteLLMProvider`/`LiteLLMClient` hold no
    per-profile state (the profile is resolved per-request from
    `InferenceRequest.model_alias`), so a single circuit breaker/retry
    policy pair covers all profiles. This is intentional - one upstream
    (LiteLLM/vLLM) backs every profile in `configs/model.yaml` today.
    """

    settings = get_settings()
    return LiteLLMProvider(
        _get_litellm_client(),
        RetryExecutor(RetryPolicy(max_retries=settings.max_retries, timeout_seconds=settings.litellm_timeout_seconds)),
        InferenceObserver(),
        CircuitBreaker(CircuitBreakerConfig()),
    )


@lru_cache(maxsize=16)
def _get_erp_inference_service(profile_name: str) -> ERPInferenceService:
    """Cached per-profile ERP inference service.

    Safe to cache: `ERPInferenceService`/`PlannerService`/
    `ResponseGenerationService` never mutate shared instance state after
    construction (unlike the general-chat path below).
    """

    profile = get_model_registry().get(profile_name)
    return ERPInferenceService(
        profile_name,
        profile,
        _get_litellm_provider(),
        get_metrics_service(),
        PromptBuilder(profile.prompts.generator or profile.prompts.planner or ''),
        _get_response_formatter(),
    )


def _build_general_inference_service(profile_name: str) -> GeneralInferenceService:
    """Fresh (uncached) per-profile general inference service.

    Deliberately NOT cached: `SecurityClassifierService.classify()` and
    `GeneralChatService.chat()` both reach into `inference_service.
    _prompt_builder` and overwrite it with their own prompt before calling
    `generate()`. If the two services shared one cached `GeneralInferenceService`
    instance (both route to the "general" profile by default in
    configs/model.yaml), concurrent security-classification and general-chat
    requests would race on that shared mutable attribute and could use the
    wrong system prompt. Constructing a fresh instance per consumer avoids
    the race entirely; the constructor itself does no I/O, so this is cheap.
    """

    profile = get_model_registry().get(profile_name)
    return GeneralInferenceService(
        profile_name,
        profile,
        _get_litellm_provider(),
        get_metrics_service(),
        PromptBuilder(profile.prompts.chat or profile.prompts.security or ''),
        _get_response_formatter(),
    )


def get_generation_service() -> ResponseGenerationService:
    profile_name, _ = get_model_registry().generation_profile()
    return ResponseGenerationService(_get_erp_inference_service(profile_name))


def get_planner_service() -> PlannerService:
    profile_name, profile = get_model_registry().planner_profile()
    return PlannerService(profile_name, profile, _get_prompt_registry(), _get_erp_inference_service(profile_name))


def get_security_classifier_service() -> SecurityClassifierService:
    profile_name, profile = get_model_registry().security_profile()
    return SecurityClassifierService(profile_name, profile, _get_prompt_registry(), _build_general_inference_service(profile_name))


def get_general_chat_service() -> GeneralChatService:
    profile_name, profile = get_model_registry().general_chat_profile()
    return GeneralChatService(profile_name, profile, _get_prompt_registry(), _build_general_inference_service(profile_name))


def get_general_inference_service() -> GeneralInferenceService:
    profile_name, _ = get_model_registry().general_chat_profile()
    return _build_general_inference_service(profile_name)


def get_health_service() -> HealthService:
    return HealthService(_get_litellm_provider())
