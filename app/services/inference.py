"""Inference service for generated text."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from app.config.settings import ModelGatewayConfig
from app.schemas.inference import GenerateRequest, GenerateResponse
from app.services.chat_template import ChatTemplateHandler
from app.services.generation_parameters import GenerationParameters
from app.services.metrics_service import MetricsService
from app.services.model_loader import ModelLoader
from app.services.prompt_builder import PromptBuilder
from app.services.response_formatter import ResponseFormatter
from app.utils.system import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceTimings:
    tokenization_ms: float
    model_generate_ms: float
    decoding_ms: float


@dataclass(frozen=True)
class InferenceResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    timings: InferenceTimings


@dataclass(frozen=True)
class GeneratePipelineResult:
    inference: InferenceResult
    prompt_length_chars: int
    prompt_source: str
    used_chat_template: bool
    prompt_preprocessing_ms: float
    prompt_building_ms: float


class InferenceService:
    """Run thread-safe model inference for text generation."""

    def __init__(
        self,
        config: ModelGatewayConfig,
        model_loader: ModelLoader,
        metrics_service: MetricsService,
        max_concurrent_requests: int,
        prompt_builder: PromptBuilder | None = None,
        chat_template_handler: ChatTemplateHandler | None = None,
        response_formatter: ResponseFormatter | None = None,
    ) -> None:
        self._config = config
        self._model_loader = model_loader
        self._metrics_service = metrics_service
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._prompt_builder = prompt_builder or PromptBuilder(config.generate_system_prompt)
        self._chat_template_handler = chat_template_handler or ChatTemplateHandler()
        self._response_formatter = response_formatter or ResponseFormatter()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        pipeline_result = await self._generate_response_text(request)
        result = pipeline_result.inference
        self._metrics_service.record_generation(result.elapsed_ms)
        response_formatting_started = time.perf_counter()
        response = self._response_formatter.format_generate_response(
            text=result.text,
            adapter=request.adapter,
            model=self._config.base_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            elapsed_ms=result.elapsed_ms,
        )
        response_formatting_ms = round((time.perf_counter() - response_formatting_started) * 1000, 2)
        logger.info(
            'generation_completed',
            extra={
                'adapter': request.adapter,
                'model': self._config.base_model,
                'prompt_source': pipeline_result.prompt_source,
                'prompt_length_chars': pipeline_result.prompt_length_chars,
                'prompt_tokens': result.prompt_tokens,
                'generation_time_ms': result.elapsed_ms,
                'output_tokens': result.completion_tokens,
                'used_chat_template': pipeline_result.used_chat_template,
                'prompt_preprocessing_ms': pipeline_result.prompt_preprocessing_ms,
                'prompt_building_ms': pipeline_result.prompt_building_ms,
                'tokenization_ms': result.timings.tokenization_ms,
                'model_generate_ms': result.timings.model_generate_ms,
                'decoding_ms': result.timings.decoding_ms,
                'response_formatting_ms': response_formatting_ms,
                'memory_usage_mb': SystemMetrics.memory_snapshot().process_memory_rss_mb,
            },
        )
        return response

    async def planner_generate(self, adapter: str, prompt: str) -> tuple[str, float]:
        result = await self._generate_text(adapter, prompt, planner_mode=True)
        self._metrics_service.record_planner(result.elapsed_ms)
        logger.info(
            'planner_generation_completed',
            extra={
                'adapter': adapter,
                'model': self._config.base_model,
                'prompt_tokens': result.prompt_tokens,
                'generation_time_ms': result.elapsed_ms,
                'output_tokens': result.completion_tokens,
                'tokenization_ms': result.timings.tokenization_ms,
                'model_generate_ms': result.timings.model_generate_ms,
                'decoding_ms': result.timings.decoding_ms,
                'memory_usage_mb': SystemMetrics.memory_snapshot().process_memory_rss_mb,
            },
        )
        return result.text, result.elapsed_ms

    async def _generate_response_text(self, request: GenerateRequest) -> GeneratePipelineResult:
        async with self._semaphore:
            model, tokenizer = await self._model_loader.get_inference_objects(request.adapter)
            prompt_bundle = self._prompt_builder.build(request)
            prompt_rendering_started = time.perf_counter()
            rendered_prompt = self._chat_template_handler.render(
                tokenizer,
                prompt_bundle.messages,
                use_chat_template=self._config.generation.use_chat_template,
            )
            prompt_building_ms = prompt_bundle.building_ms + ((time.perf_counter() - prompt_rendering_started) * 1000)
            result = await self._invoke_model(model, tokenizer, rendered_prompt.text, planner_mode=False)
            return GeneratePipelineResult(
                inference=result,
                prompt_length_chars=len(rendered_prompt.text),
                prompt_source=prompt_bundle.source,
                used_chat_template=rendered_prompt.used_chat_template,
                prompt_preprocessing_ms=prompt_bundle.preprocessing_ms,
                prompt_building_ms=round(prompt_building_ms, 2),
            )

    async def _generate_text(self, adapter: str, prompt: str, planner_mode: bool) -> InferenceResult:
        async with self._semaphore:
            model, tokenizer = await self._model_loader.get_inference_objects(adapter)
            return await self._invoke_model(model, tokenizer, prompt, planner_mode=planner_mode)

    async def _invoke_model(self, model, tokenizer: PreTrainedTokenizerBase, prompt: str, planner_mode: bool) -> InferenceResult:
        start = time.perf_counter()
        tokenization_started = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors='pt')
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generation_parameters = GenerationParameters.from_config(self._config.generation, planner_mode=planner_mode)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = await asyncio.to_thread(
                model.generate,
                **inputs,
                **generation_parameters.to_model_kwargs(tokenizer),
            )
        model_generate_ms = (time.perf_counter() - generation_started) * 1000
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._build_result(
            output,
            tokenizer,
            inputs['input_ids'].shape[1],
            elapsed_ms,
            tokenization_ms,
            model_generate_ms,
        )

    @staticmethod
    def _build_result(
        output: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        prompt_length: int,
        elapsed_ms: float,
        tokenization_ms: float,
        model_generate_ms: float,
    ) -> InferenceResult:
        sequence = output[0]
        generated_tokens = sequence[prompt_length:]
        decoding_started = time.perf_counter()
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        decoding_ms = (time.perf_counter() - decoding_started) * 1000
        completion_tokens = int(generated_tokens.shape[0])
        total_tokens = int(sequence.shape[0])
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_length,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_ms=round(elapsed_ms, 2),
            timings=InferenceTimings(
                tokenization_ms=round(tokenization_ms, 2),
                model_generate_ms=round(model_generate_ms, 2),
                decoding_ms=round(decoding_ms, 2),
            ),
        )
