"""Metrics service for the model gateway."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from app.schemas.health import MetricsResponse
from app.utils.system import SystemMetrics


@dataclass
class _MetricsState:
    requests_total: int = 0
    generation_requests: int = 0
    planner_requests: int = 0
    adapter_reload_requests: int = 0
    adapter_cache_hits: int = 0
    adapter_cache_misses: int = 0
    failures_total: int = 0
    total_generation_time_ms: float = 0.0
    last_generation_time_ms: float = 0.0


class MetricsService:
    def __init__(self) -> None:
        self._state = _MetricsState()
        self._lock = Lock()

    def record_generation(self, elapsed_ms: float) -> None:
        with self._lock:
            self._state.requests_total += 1
            self._state.generation_requests += 1
            self._state.total_generation_time_ms += elapsed_ms
            self._state.last_generation_time_ms = elapsed_ms

    def record_planner(self, elapsed_ms: float) -> None:
        with self._lock:
            self._state.requests_total += 1
            self._state.planner_requests += 1
            self._state.total_generation_time_ms += elapsed_ms
            self._state.last_generation_time_ms = elapsed_ms

    def record_adapter_reload(self) -> None:
        with self._lock:
            self._state.requests_total += 1
            self._state.adapter_reload_requests += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._state.adapter_cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._state.adapter_cache_misses += 1

    def record_failure(self) -> None:
        with self._lock:
            self._state.failures_total += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            avg = 0.0
            total_ops = self._state.generation_requests + self._state.planner_requests
            if total_ops:
                avg = round(self._state.total_generation_time_ms / total_ops, 2)
            memory = SystemMetrics.memory_snapshot()
            return MetricsResponse(
                requestsTotal=self._state.requests_total,
                generationRequests=self._state.generation_requests,
                plannerRequests=self._state.planner_requests,
                adapterReloadRequests=self._state.adapter_reload_requests,
                adapterCacheHits=self._state.adapter_cache_hits,
                adapterCacheMisses=self._state.adapter_cache_misses,
                failuresTotal=self._state.failures_total,
                avgGenerationTimeMs=avg,
                lastGenerationTimeMs=round(self._state.last_generation_time_ms, 2),
                gpuMemoryAllocatedMb=memory.gpu_memory_allocated_mb,
                processMemoryRssMb=memory.process_memory_rss_mb,
            )
