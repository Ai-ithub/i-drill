from fastapi import APIRouter, HTTPException, Query

from api.models.schemas import (
    RLAction,
    RLStateResponse,
    RLConfigResponse,
    RLHistoryResponse,
    RLResetRequest,
    RLEnvironmentState,
    RLPolicyLoadRequest,
    RLPolicyModeRequest,
    RLPolicyStatusResponse,
    RLPolicyStatus,
    RLPolicySource,
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


@router.post("/auto-step", response_model=RLStateResponse)
async def auto_step_rl_environment():
    result = rl_service.auto_step()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))

    state_payload = result.get("state") or rl_service.get_state()
    return RLStateResponse(
        success=True,
        state=RLEnvironmentState(**state_payload),
        message=result.get("message"),
    )


@router.post("/policy/load", response_model=RLPolicyStatusResponse)
async def load_rl_policy(request: RLPolicyLoadRequest):
    if request.source == RLPolicySource.MLFLOW:
        outcome = rl_service.load_policy_from_mlflow(request.model_name or "", request.stage or "Production")
    else:
        outcome = rl_service.load_policy_from_file(request.file_path or "")

    if not outcome.get("success"):
        raise HTTPException(status_code=400, detail=outcome.get("message"))

    status = RLPolicyStatus(**rl_service.get_policy_status())
    return RLPolicyStatusResponse(success=True, status=status, message=outcome.get("message"))


@router.post("/policy/mode", response_model=RLPolicyStatusResponse)
async def set_rl_policy_mode(request: RLPolicyModeRequest):
    outcome = rl_service.set_policy_mode(request.mode.value, request.auto_interval_seconds)
    if not outcome.get("success"):
        raise HTTPException(status_code=400, detail=outcome.get("message"))

    status_dict = outcome.get("status") or rl_service.get_policy_status()
    return RLPolicyStatusResponse(
        success=True,
        status=RLPolicyStatus(**status_dict),
        message=outcome.get("message"),
    )


@router.get("/policy/status", response_model=RLPolicyStatusResponse)
async def get_rl_policy_status():
    status = RLPolicyStatus(**rl_service.get_policy_status())
    return RLPolicyStatusResponse(success=True, status=status)
