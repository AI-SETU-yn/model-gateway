"""Runtime and system metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    process_memory_rss_mb: float
    gpu_memory_allocated_mb: float


class SystemMetrics:
    @staticmethod
    def memory_snapshot() -> MemorySnapshot:
        process_memory_rss_mb = 0.0
        gpu_memory_allocated_mb = 0.0

        try:
            import psutil

            process = psutil.Process()
            process_memory_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            process_memory_rss_mb = 0.0

        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory_allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            gpu_memory_allocated_mb = 0.0

        return MemorySnapshot(
            process_memory_rss_mb=process_memory_rss_mb,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
        )
