"""Inference service for generated text."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from app.config.settings import ModelGatewayConfig
from app.schemas.inference import GenerateRequest, GenerateResponse, UsageResponse
from app.services.metrics_service import MetricsService
from app.services.model_loader import ModelLoader
from app.utils.system import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float


class InferenceService:
    """Run thread-safe model inference for text generation."""

    def __init__(self, config: ModelGatewayConfig, model_loader: ModelLoader, metrics_service: MetricsService, max_concurrent_requests: int) -> None:
        self._config = config
        self._model_loader = model_loader
        self._metrics_service = metrics_service
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        result = await self._generate_text(request.adapter, request.prompt, planner_mode=False)
        self._metrics_service.record_generation(result.elapsed_ms)
        logger.info(
            'generation_completed',
            extra={
                'adapter': request.adapter,
                'generation_time_ms': result.elapsed_ms,
                'tokens_generated': result.completion_tokens,
                'memory_usage_mb': SystemMetrics.memory_snapshot().process_memory_rss_mb,
            },
        )
        return GenerateResponse(
            response=result.text,
            adapter=request.adapter,
            model=self._config.base_model,
            usage=UsageResponse(
                promptTokens=result.prompt_tokens,
                completionTokens=result.completion_tokens,
                totalTokens=result.total_tokens,
            ),
            generationTimeMs=round(result.elapsed_ms, 2),
        )

    async def planner_generate(self, adapter: str, prompt: str) -> tuple[str, float]:
        result = await self._generate_text(adapter, prompt, planner_mode=True)
        self._metrics_service.record_planner(result.elapsed_ms)
        return result.text, result.elapsed_ms

    async def _generate_text(self, adapter: str, prompt: str, planner_mode: bool) -> InferenceResult:
        async with self._semaphore:
            model, tokenizer = await self._model_loader.get_inference_objects(adapter)
            start = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            generation = self._config.generation
            max_new_tokens = generation.planner_max_new_tokens if planner_mode else generation.max_new_tokens
            with torch.inference_mode():
                output = await asyncio.to_thread(
                    model.generate,
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    do_sample=generation.do_sample,
                    repetition_penalty=generation.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._build_result(output, tokenizer, inputs['input_ids'].shape[1], elapsed_ms)

    @staticmethod
    def _build_result(output: torch.Tensor, tokenizer: PreTrainedTokenizerBase, prompt_length: int, elapsed_ms: float) -> InferenceResult:
        sequence = output[0]
        generated_tokens = sequence[prompt_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(generated_tokens.shape[0])
        total_tokens = int(sequence.shape[0])
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_length,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_ms=round(elapsed_ms, 2),
        )
