from __future__ import annotations

import time
from dataclasses import dataclass

from app.config.settings import ModelProfileConfig
from app.models.inference import InferenceRequest
from app.providers.base import BaseInferenceProvider
from app.schemas.inference import ModelHealthItem, ModelHealthResponse


@dataclass(frozen=True)
class HealthProbeResult:
    profile: str
    provider: str
    model: str
    base_url: str
    adapter_enabled: bool
    adapter: str | None
    status: str
    latency_ms: float
    last_checked: str


class HealthService:
    def __init__(self, provider: BaseInferenceProvider) -> None:
        self._provider = provider

    async def check_profile(self, profile_name: str, profile: ModelProfileConfig) -> HealthProbeResult:
        started = time.perf_counter()
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        adapter = profile.default_adapter if profile.adapter_enabled else None
        request = InferenceRequest(
            modelAlias=profile_name,
            systemPrompt='Health check',
            userPrompt='Return ok',
            adapter=adapter,
            maxTokens=1,
            temperature=0.0,
            top_p=1.0,
            metadata={
                'provider': profile.provider,
                'model_name': profile.model_name,
                'api_base': profile.base_url,
                'api_key': profile.api_key,
                'request_id': f'health-{profile_name}',
                'trace_id': f'health-{profile_name}',
            },
        )
        status = 'UP'
        try:
            await self._provider.complete(request)
        except Exception:
            status = 'DOWN'
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return HealthProbeResult(
            profile=profile_name,
            provider='LiteLLM',
            model=profile.model_name,
            base_url=profile.base_url,
            adapter_enabled=profile.adapter_enabled,
            adapter=adapter,
            status=status,
            latency_ms=latency_ms,
            last_checked=timestamp,
        )

    async def check_all(self, profiles: dict[str, ModelProfileConfig]) -> ModelHealthResponse:
        results = []
        for profile_name, profile in profiles.items():
            probe = await self.check_profile(profile_name, profile)
            results.append(
                ModelHealthItem(
                    profile=probe.profile,
                    model=probe.model,
                    provider=probe.provider,
                    baseUrl=probe.base_url,
                    status=probe.status,
                    latencyMs=probe.latency_ms,
                    adapterEnabled=probe.adapter_enabled,
                    adapter=probe.adapter,
                    lastChecked=probe.last_checked,
                )
            )
        return ModelHealthResponse(models=results)
