"""Adapter discovery and cache management."""

from __future__ import annotations

from pathlib import Path

from app.exceptions.errors import AdapterNotFoundError, InvalidAdapterError


class AdapterManager:
    """Resolve adapter paths and track discovered adapter folders."""

    def __init__(self, project_root: Path, adapters_root: Path) -> None:
        self._adapters_root = (project_root / adapters_root).resolve()
        self._adapters_root.mkdir(parents=True, exist_ok=True)

    @property
    def adapters_root(self) -> Path:
        return self._adapters_root

    def get_adapter_path(self, adapter_name: str) -> Path:
        self._validate_adapter_name(adapter_name)
        adapter_path = (self._adapters_root / adapter_name).resolve()
        if not adapter_path.exists() or not adapter_path.is_dir():
            raise AdapterNotFoundError(f'Adapter "{adapter_name}" was not found on disk.')
        if not (adapter_path / 'adapter_config.json').exists() or not (adapter_path / 'adapter_model.safetensors').exists():
            raise InvalidAdapterError(f'Adapter "{adapter_name}" is missing required adapter files.')
        return adapter_path

    def list_adapters(self) -> list[str]:
        return sorted(path.name for path in self._adapters_root.iterdir() if path.is_dir())

    @staticmethod
    def _validate_adapter_name(adapter_name: str) -> None:
        if not adapter_name or any(char in adapter_name for char in ('..', '/', '\\')):
            raise InvalidAdapterError('Adapter name is invalid.')
