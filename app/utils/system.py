"""Runtime and system metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True)
class MemorySnapshot:
    process_memory_rss_mb: float
    gpu_memory_allocated_mb: float


class SystemMetrics:
    @staticmethod
    def memory_snapshot() -> MemorySnapshot:
        process = psutil.Process()
        process_memory_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        gpu_memory_allocated_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return MemorySnapshot(
            process_memory_rss_mb=process_memory_rss_mb,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
        )
