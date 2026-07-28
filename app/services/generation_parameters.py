"""Reusable model generation parameter mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from app.config.settings import GenerationConfig


@dataclass(frozen=True)
class GenerationParameters:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float

    @classmethod
    def from_config(cls, config: GenerationConfig, *, planner_mode: bool) -> 'GenerationParameters':
        return cls(
            max_new_tokens=config.planner_max_new_tokens if planner_mode else config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
        )

    def to_model_kwargs(self, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        return {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': self.do_sample,
            'repetition_penalty': self.repetition_penalty,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
