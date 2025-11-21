"""
Real-time DVR API Routes
For real-time data validation, reconciliation, and alerts
"""
from fastapi import APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import logging
import json

from services.realtime_dvr_service import realtime_dvr_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dvr/realtime", tags=["Real-time DVR"])


# Request Models
class ValidationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Sensor data to validate")
    previous_data: Optional[Dict[str, Any]] = Field(None, description="Previous data for comparison")


class ReconciliationRequest(BaseModel):
    surface_data: Dict[str, Any] = Field(..., description="Surface sensor data")
    lwd_data: Optional[Dict[str, Any]] = Field(None, description="LWD data for reconciliation")


@router.post("/validate", response_model=Dict[str, Any])
async def validate_realtime(
    request: ValidationRequest,
    current_user: User = Depends(get_current_user)
):
    """Validate sensor data in real-time."""
    try:
        result = realtime_dvr_service.validate_realtime(
            request.data,
            request.previous_data
        )
        return result
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/reconcile", response_model=Dict[str, Any])
async def reconcile_realtime(
    request: ReconciliationRequest,
    current_user: User = Depends(get_current_user)
):
    """Reconcile surface and LWD data in real-time."""
    try:
        reconciled = realtime_dvr_service.reconcile_realtime(
            request.surface_data,
            request.lwd_data
        )
        return {
            "success": True,
            "reconciled_data": reconciled
        }
    except Exception as e:
        logger.error(f"Error reconciling data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/process", response_model=Dict[str, Any])
async def process_and_alert(
    data: Dict[str, Any],
    lwd_data: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """Process data through real-time DVR and send alerts if needed."""
    try:
        result = await realtime_dvr_service.process_and_alert(data, lwd_data)
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/parameter-ranges", response_model=Dict[str, Any])
async def get_parameter_ranges(
    current_user: User = Depends(get_current_user)
):
    """Get parameter ranges for validation."""
    return {
        "parameter_ranges": realtime_dvr_service.parameter_ranges,
        "rate_of_change_thresholds": realtime_dvr_service.rate_of_change_thresholds
    }


@router.put("/parameter-ranges", response_model=Dict[str, Any])
async def update_parameter_ranges(
    parameter_ranges: Optional[Dict[str, tuple]] = None,
    rate_of_change_thresholds: Optional[Dict[str, float]] = None,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Update parameter ranges and thresholds."""
    try:
        if parameter_ranges:
            realtime_dvr_service.parameter_ranges.update(parameter_ranges)
        if rate_of_change_thresholds:
            realtime_dvr_service.rate_of_change_thresholds.update(rate_of_change_thresholds)
        
        return {
            "success": True,
            "message": "Parameter ranges updated",
            "parameter_ranges": realtime_dvr_service.parameter_ranges,
            "rate_of_change_thresholds": realtime_dvr_service.rate_of_change_thresholds
        }
    except Exception as e:
        logger.error(f"Error updating parameter ranges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.websocket("/alerts/{rig_id}")
async def dvr_alerts_websocket(websocket: WebSocket, rig_id: str):
    """WebSocket endpoint for real-time DVR alerts."""
    await websocket.accept()
    logger.info(f"DVR alerts WebSocket connected for rig {rig_id}")
    
    try:
        while True:
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Echo back or process message
                await websocket.send_json({
                    "type": "ack",
                    "message": "Received"
                })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    except WebSocketDisconnect:
        logger.info(f"DVR alerts WebSocket disconnected for rig {rig_id}")
    except Exception as e:
        logger.error(f"Error in DVR alerts WebSocket: {e}")
        await websocket.close()


@router.get("/history/{rig_id}", response_model=Dict[str, Any])
async def get_validation_history(
    rig_id: str,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get validation history for a rig."""
    try:
        from services.dvr_service import dvr_service
        history = dvr_service.get_history(limit=limit, rig_id=rig_id)
        return {
            "rig_id": rig_id,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting validation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

