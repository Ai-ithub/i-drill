from fastapi import APIRouter, HTTPException, Query

from api.models.schemas import (
    RLAction,
    RLStateResponse,
    RLConfigResponse,
    RLHistoryResponse,
    RLResetRequest,
    RLEnvironmentState,
)
from services.rl_service import rl_service

router = APIRouter(prefix="/rl", tags=["rl"])


@router.get("/config", response_model=RLConfigResponse)
async def get_rl_config():
    config = rl_service.get_config()
    return RLConfigResponse(success=True, config=config)


@router.get("/state", response_model=RLStateResponse)
async def get_rl_state():
    state = rl_service.get_state()
    return RLStateResponse(success=True, state=RLEnvironmentState(**state))


@router.post("/reset", response_model=RLStateResponse)
async def reset_rl_environment(request: RLResetRequest):
    state = rl_service.reset(random_init=request.random_init)
    return RLStateResponse(success=True, state=RLEnvironmentState(**state))


@router.post("/step", response_model=RLStateResponse)
async def step_rl_environment(action: RLAction):
    try:
        state = rl_service.step(action.model_dump())
        return RLStateResponse(success=True, state=RLEnvironmentState(**state))
    except Exception as exc:  # pragma: no cover - unexpected errors logged
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history", response_model=RLHistoryResponse)
async def get_rl_history(limit: int = Query(50, ge=1, le=1000)):
    history = [RLEnvironmentState(**entry) for entry in rl_service.get_history(limit)]
    return RLHistoryResponse(success=True, history=history)
