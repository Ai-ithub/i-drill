"""
Integration API Routes
Handles integrated operations between RL Models and DVR
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from api.models.schemas import (
    SensorDataPoint,
    RLAction,
)
from services.integration_service import integration_service

router = APIRouter(prefix="/integration", tags=["integration"])


@router.post("/sensor-to-rl")
async def process_sensor_data_for_rl(
    sensor_data: SensorDataPoint,
    apply_to_rl: bool = Query(False, description="Apply validated data to RL environment")
):
    """
    Process sensor data through DVR and optionally feed to RL.
    
    Pipeline:
    1. Validate and reconcile sensor data using DVR
    2. If valid and apply_to_rl=True, feed to RL environment
    3. Return integrated processing result
    """
    try:
        result = integration_service.process_sensor_data_for_rl(
            sensor_data.model_dump(),
            apply_to_rl=apply_to_rl
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Processing failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rl-action-with-dvr")
async def process_rl_action_with_dvr(
    action: RLAction,
    validate_with_dvr: bool = Query(True, description="Validate action through DVR"),
    history_size: int = Query(100, ge=10, le=1000, description="History size for DVR validation")
):
    """
    Process RL action with DVR validation before applying.
    
    Pipeline:
    1. Validate action using DVR anomaly detection
    2. If validation passes, apply action to RL environment
    3. Return integrated result
    """
    try:
        result = integration_service.process_rl_action_with_dvr(
            action.model_dump(),
            validate_with_dvr=validate_with_dvr,
            history_size=history_size
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Action processing failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/validate-with-rl-context")
async def validate_with_rl_context(
    sensor_data: SensorDataPoint,
    use_rl_state: bool = Query(True, description="Use RL state for enhanced validation")
):
    """
    Validate sensor data using DVR with RL state context.
    
    Uses current RL environment state to provide context for validation,
    allowing for more intelligent anomaly detection.
    """
    try:
        result = integration_service.validate_with_rl_context(
            sensor_data.model_dump(),
            use_rl_state=use_rl_state
        )
        
        return result
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auto-step-integrated")
async def integrated_auto_step(
    validate_action: bool = Query(True, description="Validate action through DVR before applying")
):
    """
    Execute RL auto step with integrated DVR validation.
    
    Pipeline:
    1. Get action from RL policy (auto step)
    2. Validate action through DVR
    3. Apply action to environment
    4. Return integrated result
    """
    try:
        result = integration_service.integrated_auto_step(
            validate_action=validate_action
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Auto step failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_integration_status():
    """
    Get status of RL-DVR integration.
    
    Returns information about:
    - RL availability and policy status
    - DVR availability
    - Integration active status
    - Current RL episode and step
    """
    try:
        status = integration_service.get_integration_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

