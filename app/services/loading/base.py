"""Shared model loader abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

PreTrainedTokenizerBase = Any


class BaseModelLoader(ABC):
    @property
    @abstractmethod
    def profile_name(self) -> str: ...

    @property
    @abstractmethod
    def configured_model_name(self) -> str: ...

    @property
    @abstractmethod
    def resolved_device(self) -> str: ...

    @property
    @abstractmethod
    def resolved_dtype(self) -> str: ...

    @property
    @abstractmethod
    def active_adapter(self) -> str | None: ...

    @property
    @abstractmethod
    def loaded_adapters(self) -> list[str]: ...

    @property
    @abstractmethod
    def base_model_loaded(self) -> bool: ...

    @abstractmethod
    async def initialize(self, preload_default_adapter: bool = False) -> None: ...

    @abstractmethod
    async def get_inference_objects(self, adapter_name: str | None = None) -> tuple[object, PreTrainedTokenizerBase]: ...

    @abstractmethod
    async def reload_adapter(self, adapter_name: str) -> None: ...

    @abstractmethod
    def adapter_location(self, adapter_name: str) -> Path | None: ...
