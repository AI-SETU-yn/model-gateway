from __future__ import annotations

import time
from collections.abc import AsyncIterator

import torch

from app.models.inference import InferenceRequest, InferenceResponse, InferenceUsage
from app.providers.base import BaseInferenceProvider
from app.services.loading.base import BaseModelLoader


class TransformersProvider(BaseInferenceProvider):
    def __init__(self, loaders: dict[str, BaseModelLoader]) -> None:
        self._loaders = loaders

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        loader = self._loaders[request.model_alias]
        model, tokenizer = await loader.get_inference_objects(request.adapter)
        messages = request.to_messages()
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
        inputs = tokenizer(prompt_text, return_tensors='pt')
        target_device = getattr(model, 'device', None)
        if target_device is not None:
            inputs = {key: value.to(target_device) for key, value in inputs.items()}
        generation_kwargs = {
            'max_new_tokens': request.max_tokens or 128,
            'temperature': request.temperature if request.temperature is not None else 0.1,
            'top_p': request.top_p if request.top_p is not None else 0.9,
            'do_sample': (request.temperature or 0.0) > 0,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        prompt_tokens = int(inputs['input_ids'].shape[-1])
        completion_tokens = int(output.shape[-1] - prompt_tokens)
        generated_ids = output[0][prompt_tokens:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return InferenceResponse(
            content=content,
            finishReason='stop',
            usage=InferenceUsage(
                promptTokens=prompt_tokens,
                completionTokens=max(completion_tokens, 0),
                totalTokens=prompt_tokens + max(completion_tokens, 0),
            ),
            latencyMs=round((time.perf_counter() - started) * 1000, 2),
            provider='transformers',
            model=loader.configured_model_name,
            adapter=request.adapter,
            requestId=str(request.metadata.get('request_id') or ''),
            traceId=str(request.metadata.get('trace_id') or ''),
            stream=request.stream,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        yield await self.complete(request.model_copy(update={'stream': True}))
