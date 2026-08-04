from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models.inference import InferenceRequest, InferenceResponse


class BaseInferenceClient(ABC):
    @abstractmethod
    async def completion(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream_completion(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        raise NotImplementedError
