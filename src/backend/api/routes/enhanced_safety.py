"""
Enhanced Safety API Routes
Improved safety operations with automatic shutdown, predictive analysis, and enhanced alerts
"""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import logging

from services.enhanced_safety_service import enhanced_safety_service
from services.safety_service import safety_service
from services.drilling_events_service import drilling_events_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User, SensorDataPoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/safety/enhanced", tags=["Enhanced Safety"])


# Request Models
class EmergencyStopDashboardRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    reason: str = Field(..., description="Reason for emergency stop")
    description: Optional[str] = Field(None, description="Additional description")


class AutoShutdownConfigRequest(BaseModel):
    enabled: bool = Field(True, description="Enable automatic shutdown")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold for auto-shutdown")


@router.post("/emergency-stop/dashboard", response_model=Dict[str, Any])
async def emergency_stop_from_dashboard(
    request: EmergencyStopDashboardRequest,
    current_user: User = Depends(require_role(["admin", "engineer", "driller"]))
):
    """
    Execute emergency stop from dashboard with full audit trail.
    
    This endpoint allows authorized users to trigger emergency stop from the dashboard.
    All actions are logged with full audit trail including user information.
    """
    try:
        result = enhanced_safety_service.emergency_stop_from_dashboard(
            rig_id=request.rig_id,
            reason=request.reason,
            user_id=current_user.id,
            user_name=current_user.username,
            description=request.description
        )
        
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Emergency stop failed")
            )
    except Exception as e:
        logger.error(f"Error executing emergency stop from dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/emergency-stop/audit", response_model=List[Dict[str, Any]])
async def get_emergency_stop_audit(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    current_user: User = Depends(get_current_user)
):
    """Get emergency stop audit trail."""
    try:
        history = enhanced_safety_service.get_emergency_stop_history(rig_id=rig_id, limit=limit)
        return history
    except Exception as e:
        logger.error(f"Error getting emergency stop audit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/kick/detect-enhanced", response_model=Dict[str, Any])
async def detect_kick_enhanced(
    sensor_data: SensorDataPoint,
    current_user: User = Depends(get_current_user)
):
    """Enhanced kick detection with automatic shutdown capability."""
    try:
        result = enhanced_safety_service.detect_kick_enhanced(sensor_data.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error in enhanced kick detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stuck-pipe/detect-enhanced", response_model=Dict[str, Any])
async def detect_stuck_pipe_enhanced(
    sensor_data: SensorDataPoint,
    current_user: User = Depends(get_current_user)
):
    """Enhanced stuck pipe detection with predictive analysis."""
    try:
        result = enhanced_safety_service.detect_stuck_pipe_enhanced(sensor_data.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error in enhanced stuck pipe detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/stuck-pipe/prediction/{rig_id}", response_model=Dict[str, Any])
async def get_stuck_pipe_prediction(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get stuck pipe prediction statistics for a rig."""
    try:
        prediction = enhanced_safety_service.get_stuck_pipe_prediction(rig_id)
        return prediction
    except Exception as e:
        logger.error(f"Error getting stuck pipe prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/formation-change/detect-enhanced", response_model=Dict[str, Any])
async def detect_formation_change_enhanced(
    sensor_data: SensorDataPoint,
    current_user: User = Depends(get_current_user)
):
    """Enhanced formation change detection with improved alerts."""
    try:
        result = enhanced_safety_service.detect_formation_change_enhanced(sensor_data.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error in enhanced formation change detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/auto-shutdown/config", response_model=Dict[str, Any])
async def get_auto_shutdown_config(
    current_user: User = Depends(get_current_user)
):
    """Get automatic shutdown configuration."""
    return {
        "enabled": enhanced_safety_service.kick_auto_shutdown_enabled,
        "confidence_threshold": enhanced_safety_service.kick_auto_shutdown_confidence_threshold
    }


@router.put("/auto-shutdown/config", response_model=Dict[str, Any])
async def update_auto_shutdown_config(
    request: AutoShutdownConfigRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Update automatic shutdown configuration."""
    try:
        enhanced_safety_service.kick_auto_shutdown_enabled = request.enabled
        enhanced_safety_service.kick_auto_shutdown_confidence_threshold = request.confidence_threshold
        
        logger.info(
            f"Auto-shutdown config updated: enabled={request.enabled}, "
            f"threshold={request.confidence_threshold}"
        )
        
        return {
            "success": True,
            "message": "Auto-shutdown configuration updated",
            "config": {
                "enabled": request.enabled,
                "confidence_threshold": request.confidence_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error updating auto-shutdown config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_safety_monitoring(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Start real-time safety monitoring."""
    try:
        enhanced_safety_service.start_monitoring()
        return {
            "success": True,
            "message": "Safety monitoring started"
        }
    except Exception as e:
        logger.error(f"Error starting safety monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/stop", response_model=Dict[str, Any])
async def stop_safety_monitoring(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Stop real-time safety monitoring."""
    try:
        enhanced_safety_service.stop_monitoring()
        return {
            "success": True,
            "message": "Safety monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping safety monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/status", response_model=Dict[str, Any])
async def get_monitoring_status(
    current_user: User = Depends(get_current_user)
):
    """Get safety monitoring status."""
    return {
        "running": enhanced_safety_service.running,
        "monitoring_interval": enhanced_safety_service.monitoring_interval,
        "auto_shutdown_enabled": enhanced_safety_service.kick_auto_shutdown_enabled
    }

