from abc import ABC, abstractmethod

from app.models.inference import InferenceRequest, InferenceResponse


class BaseInferenceClient(ABC):
    @abstractmethod
    async def completion(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError
