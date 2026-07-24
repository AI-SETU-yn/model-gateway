import logging

from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GatewayService:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        logger.info(
            'gateway_generate_requested',
            extra={
                'conversation_id': request.conversation_id,
                'metadata_keys': sorted(request.metadata.keys()),
            },
        )
        response = await self._provider.generate(request)
        logger.info('gateway_generate_completed', extra={'model_name': response.model})
        return response
