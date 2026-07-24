from abc import ABC, abstractmethod

from app.models.request import GenerateRequest
from app.models.response import GenerateResponse


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError
