"""Shared generation execution primitives."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.config.settings import GenerationConfig
from app.services.generation_parameters import GenerationParameters

PreTrainedTokenizerBase = Any


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


class GenerationRunner:
    async def run(
        self,
        model: object,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        generation: GenerationConfig,
        *,
        planner_mode: bool,
    ) -> InferenceResult:
        import torch

        start = time.perf_counter()
        tokenization_started = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors='pt')
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generation_parameters = GenerationParameters.from_config(generation, planner_mode=planner_mode)
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
        output,
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
