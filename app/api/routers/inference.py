from fastapi import APIRouter, Depends

from app.schemas.inference import (
    GenerateRequest,
    GenerateResponse,
    GeneralChatRequest,
    ModelHealthResponse,
    PlannerRequest,
    PlannerResponse,
    SecurityClassificationResponse,
    SecurityClassifyRequest,
)
from app.services.dependencies import (
    get_generation_service,
    get_general_chat_service,
    get_general_inference_service,
    get_model_registry,
    get_planner_service,
    get_security_classifier_service,
)
from app.services.generation.service import ResponseGenerationService
from app.services.general_chat.service import GeneralChatService
from app.services.inference import GeneralInferenceService
from app.services.planner import PlannerService
from app.services.security_classifier import SecurityClassifierService

router = APIRouter(tags=['inference'])


@router.post('/generate', response_model=GenerateResponse)
async def generate(payload: GenerateRequest, service: ResponseGenerationService = Depends(get_generation_service)) -> GenerateResponse:
    return await service.generate(payload)


@router.post('/planner', response_model=PlannerResponse)
async def planner(payload: PlannerRequest, service: PlannerService = Depends(get_planner_service)) -> PlannerResponse:
    return await service.plan(payload)


@router.post('/security/classify', response_model=SecurityClassificationResponse)
async def classify_security(payload: SecurityClassifyRequest, service: SecurityClassifierService = Depends(get_security_classifier_service)) -> SecurityClassificationResponse:
    return await service.classify(payload)


@router.post('/chat/general', response_model=GenerateResponse)
async def general_chat(payload: GeneralChatRequest, service: GeneralChatService = Depends(get_general_chat_service)) -> GenerateResponse:
    return await service.chat(payload)
