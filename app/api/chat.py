"""Chat API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.config import get_chat_service
from app.models.request import ChatRequest
from app.models.response import ChatCompletionResponse, ModelListResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=['chat'])


@router.get('/v1/models', response_model=ModelListResponse)
async def list_models(service: ChatService = Depends(get_chat_service)) -> ModelListResponse:
    """List models registered by the gateway."""

    return await service.list_models()


@router.post('/chat', response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """Generate a chat completion via LiteLLM."""

    if payload.stream:
        event_stream = await service.stream_chat(payload)
        return StreamingResponse(event_stream, media_type='text/event-stream')
    return await service.generate_chat(payload)
