"""Base model and adapter loading service."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.config.settings import ModelGatewayConfig
from app.exceptions.errors import GPUUnavailableError, ModelLoadError
from app.services.adapter_manager import AdapterManager

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load the base model once and hot-load LoRA adapters safely."""

    def __init__(self, config: ModelGatewayConfig, adapter_manager: AdapterManager) -> None:
        self._config = config
        self._adapter_manager = adapter_manager
        self._model: PeftModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._loaded_adapters: dict[str, Path] = {}
        self._active_adapter: str | None = None
        self._lock = asyncio.Lock()
        self._device_map = self._resolve_device_map(config.device)
        self._resolved_device = self._resolve_device_label(config.device)
        self._resolved_dtype = self._resolve_dtype(config.dtype)

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    @property
    def resolved_dtype(self) -> str:
        return str(self._resolved_dtype).replace('torch.', '') if self._resolved_dtype is not None else 'auto'

    @property
    def active_adapter(self) -> str | None:
        return self._active_adapter

    @property
    def loaded_adapters(self) -> list[str]:
        return sorted(self._loaded_adapters)

    @property
    def base_model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def initialize(self, preload_default_adapter: bool = False) -> None:
        await self._ensure_base_loaded()
        if preload_default_adapter:
            await self.ensure_adapter_loaded(self._config.default_adapter)

    async def _ensure_base_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        async with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            self._validate_device_request()
            try:
                model_kwargs: dict[str, Any] = {
                    'trust_remote_code': self._config.trust_remote_code,
                    'device_map': self._device_map,
                }
                if self._resolved_dtype is not None:
                    model_kwargs['torch_dtype'] = self._resolved_dtype
                base_model = AutoModelForCausalLM.from_pretrained(self._config.base_model, **model_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(self._config.base_model, use_fast=True, trust_remote_code=self._config.trust_remote_code)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                self._model = base_model
                self._tokenizer = tokenizer
                logger.info('base_model_loaded', extra={'model': self._config.base_model, 'device': self._resolved_device, 'dtype': self.resolved_dtype})
            except Exception as exc:
                raise ModelLoadError(f'Failed to load base model: {exc}') from exc

    async def get_inference_objects(self, adapter_name: str) -> tuple[PeftModel, PreTrainedTokenizerBase]:
        await self.ensure_adapter_loaded(adapter_name)
        assert self._model is not None and self._tokenizer is not None
        return self._model, self._tokenizer

    async def ensure_adapter_loaded(self, adapter_name: str) -> None:
        await self._ensure_base_loaded()
        adapter_path = self._adapter_manager.get_adapter_path(adapter_name)
        async with self._lock:
            assert self._model is not None
            if adapter_name not in self._loaded_adapters:
                try:
                    if isinstance(self._model, PeftModel):
                        self._model.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
                    else:
                        self._model = PeftModel.from_pretrained(self._model, str(adapter_path), adapter_name=adapter_name, is_trainable=False)
                    self._loaded_adapters[adapter_name] = adapter_path
                    logger.info('adapter_loaded', extra={'adapter': adapter_name, 'path': str(adapter_path)})
                except Exception as exc:
                    raise ModelLoadError(f'Failed to load adapter "{adapter_name}": {exc}') from exc
            self._model.set_adapter(adapter_name)
            self._active_adapter = adapter_name

    async def reload_adapter(self, adapter_name: str) -> None:
        await self._ensure_base_loaded()
        adapter_path = self._adapter_manager.get_adapter_path(adapter_name)
        async with self._lock:
            assert self._model is not None
            if adapter_name in self._loaded_adapters:
                try:
                    self._model.delete_adapter(adapter_name)
                except Exception:
                    logger.warning('adapter_delete_before_reload_failed', extra={'adapter': adapter_name})
                self._loaded_adapters.pop(adapter_name, None)
            try:
                if isinstance(self._model, PeftModel):
                    self._model.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
                else:
                    self._model = PeftModel.from_pretrained(self._model, str(adapter_path), adapter_name=adapter_name, is_trainable=False)
                self._model.set_adapter(adapter_name)
                self._loaded_adapters[adapter_name] = adapter_path
                self._active_adapter = adapter_name
                logger.info('adapter_reloaded', extra={'adapter': adapter_name, 'path': str(adapter_path)})
            except Exception as exc:
                raise ModelLoadError(f'Failed to reload adapter "{adapter_name}": {exc}') from exc

    @staticmethod
    def _resolve_device_map(device: str) -> str | dict[str, str] | None:
        if device == 'auto':
            return 'auto'
        if device == 'cpu':
            return {'': 'cpu'}
        if device == 'cuda':
            return {'': 'cuda:0'}
        return {'': device}

    @staticmethod
    def _resolve_device_label(device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype | None:
        mapping = {
            'auto': None,
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if dtype not in mapping:
            raise ModelLoadError(f'Unsupported dtype: {dtype}')
        return mapping[dtype]

    def _validate_device_request(self) -> None:
        if self._config.device == 'cuda' and not torch.cuda.is_available():
            raise GPUUnavailableError('CUDA device requested but no GPU is available.')
