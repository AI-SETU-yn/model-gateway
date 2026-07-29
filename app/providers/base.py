from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.models.inference import InferenceRequest, InferenceResponse


class BaseInferenceProvider(ABC):
    @abstractmethod
    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        raise NotImplementedError
