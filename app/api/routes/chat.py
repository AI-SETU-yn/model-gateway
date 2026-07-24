from fastapi import APIRouter, Depends, Request

from app.models.request import GenerateRequest, HealthResponse
from app.models.response import GenerateResponse
from app.services.dependencies import get_gateway_service
from app.services.gateway_service import GatewayService

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status='ok')


@router.get('/ready', response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    ready_flag = getattr(request.app.state, 'is_ready', False)
    return HealthResponse(status='ok' if ready_flag else 'not_ready')


@router.post('/generate', response_model=GenerateResponse)
async def generate(
    payload: GenerateRequest,
    request: Request,
    gateway_service: GatewayService = Depends(get_gateway_service),
) -> GenerateResponse:
    if payload.conversation_id and not request.state.conversation_id:
        request.state.conversation_id = payload.conversation_id
    return await gateway_service.generate(payload)
