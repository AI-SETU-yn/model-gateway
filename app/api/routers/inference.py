from fastapi import APIRouter, Depends

from app.schemas.inference import GenerateRequest, GenerateResponse, PlannerRequest, PlannerResponse
from app.services.dependencies import get_inference_service, get_planner_service
from app.services.inference import InferenceService
from app.services.planner import PlannerService

router = APIRouter(tags=['inference'])


@router.post('/generate', response_model=GenerateResponse)
async def generate(payload: GenerateRequest, service: InferenceService = Depends(get_inference_service)) -> GenerateResponse:
    return await service.generate(payload)


@router.post('/planner', response_model=PlannerResponse)
async def planner(payload: PlannerRequest, service: PlannerService = Depends(get_planner_service)) -> PlannerResponse:
    return await service.plan(payload)
