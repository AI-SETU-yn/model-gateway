from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    status: str
    base_model_loaded: bool = Field(alias='baseModelLoaded')
    default_adapter: str = Field(alias='defaultAdapter')
    loaded_adapters: list[str] = Field(alias='loadedAdapters')
    device: str
    dtype: str


class ReloadAdapterResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: str
    reloaded: bool
    loaded_adapters: list[str] = Field(alias='loadedAdapters')


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    requests_total: int = Field(alias='requestsTotal')
    generation_requests: int = Field(alias='generationRequests')
    planner_requests: int = Field(alias='plannerRequests')
    adapter_reload_requests: int = Field(alias='adapterReloadRequests')
    adapter_cache_hits: int = Field(alias='adapterCacheHits')
    adapter_cache_misses: int = Field(alias='adapterCacheMisses')
    failures_total: int = Field(alias='failuresTotal')
    avg_generation_time_ms: float = Field(alias='avgGenerationTimeMs')
    last_generation_time_ms: float = Field(alias='lastGenerationTimeMs')
    gpu_memory_allocated_mb: float = Field(alias='gpuMemoryAllocatedMb')
    process_memory_rss_mb: float = Field(alias='processMemoryRssMb')


class AdapterInventoryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapters: list[str]
