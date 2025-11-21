"""
Safety API Routes
Handles emergency stop, kick detection, stuck pipe detection, and other safety operations
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from datetime import datetime
import logging

from api.models.schemas import (
    EmergencyStopRequest,
    EmergencyStopResponse,
    KickDetectionResponse,
    StuckPipeDetectionResponse,
    SafetyEventResponse,
    SensorDataPoint
)
from api.models.database_models import UserDB, SafetyEventDB
from api.dependencies import get_current_active_user, get_current_engineer_user
from database import db_manager
from services.safety_service import safety_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/safety", tags=["safety"])


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop(
    request: EmergencyStopRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> EmergencyStopResponse:
    """
    Execute emergency stop for drilling operations
    
    This endpoint immediately stops all drilling operations for the specified rig.
    It performs the following actions:
    1. Stops RPM and reduces WOB to 0
    2. Maintains mud flow for well control
    3. Broadcasts emergency stop to all connected clients
    4. Creates a safety event record
    
    **Critical Operation**: This action requires immediate attention and cannot be undone automatically.
    
    Args:
        request: Emergency stop request with rig_id and reason
        current_user: Authenticated user making the request
        
    Returns:
        EmergencyStopResponse with success status and actions taken
        
    Raises:
        HTTPException: If emergency stop fails
    """
    try:
        result = safety_service.emergency_stop(
            rig_id=request.rig_id,
            reason=request.reason,
            description=request.description,
            user_id=current_user.id
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Emergency stop failed")
            )
        
        return EmergencyStopResponse(
            success=True,
            message=result.get("message", "Emergency stop executed successfully"),
            event_id=result.get("event_id"),
            timestamp=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat())),
            actions_taken=result.get("actions_taken", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute emergency stop: {str(e)}"
        )


@router.post("/detect-kick", response_model=KickDetectionResponse)
async def detect_kick(
    sensor_data: SensorDataPoint,
    current_user: UserDB = Depends(get_current_active_user)
) -> KickDetectionResponse:
    """
    Detect kick (gas influx) based on sensor data
    
    Analyzes current sensor data to detect potential kick conditions:
    - Flow differential (flow_out - flow_in)
    - Pit volume changes
    - Standpipe pressure changes
    
    Args:
        sensor_data: Current sensor data point
        current_user: Authenticated user
        
    Returns:
        KickDetectionResponse with detection result and recommendations
    """
    try:
        result = safety_service.detect_kick(sensor_data.model_dump())
        
        return KickDetectionResponse(
            kick_detected=result.get("kick_detected", False),
            severity=result.get("severity", "low"),
            confidence=result.get("confidence", 0.0),
            indicators=result.get("indicators", {}),
            immediate_actions=result.get("immediate_actions", []),
            event_id=result.get("event_id"),
            timestamp=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat()))
        )
        
    except Exception as e:
        logger.error(f"Error detecting kick: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect kick: {str(e)}"
        )


@router.post("/detect-stuck-pipe", response_model=StuckPipeDetectionResponse)
async def detect_stuck_pipe(
    sensor_data: SensorDataPoint,
    current_user: UserDB = Depends(get_current_active_user)
) -> StuckPipeDetectionResponse:
    """
    Detect stuck pipe condition
    
    Analyzes sensor data to detect potential stuck pipe conditions:
    - ROP decrease
    - Torque increase
    - Hook load decrease
    - Vibration spikes
    
    Args:
        sensor_data: Current sensor data point
        current_user: Authenticated user
        
    Returns:
        StuckPipeDetectionResponse with detection result and recommendations
    """
    try:
        result = safety_service.detect_stuck_pipe(sensor_data.model_dump())
        
        return StuckPipeDetectionResponse(
            stuck_pipe_detected=result.get("stuck_pipe_detected", False),
            risk_level=result.get("risk_level", "low"),
            risk_score=result.get("risk_score", 0.0),
            indicators=result.get("indicators", {}),
            recommended_actions=result.get("recommended_actions", []),
            event_id=result.get("event_id"),
            timestamp=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat()))
        )
        
    except Exception as e:
        logger.error(f"Error detecting stuck pipe: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect stuck pipe: {str(e)}"
        )


@router.get("/events", response_model=List[SafetyEventResponse])
async def get_safety_events(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: UserDB = Depends(get_current_active_user)
) -> List[SafetyEventResponse]:
    """
    Get safety events with optional filtering
    
    Returns a list of safety events (emergency stops, kicks, stuck pipe, etc.)
    with optional filtering by rig_id, event_type, severity, and status.
    
    Args:
        rig_id: Optional rig ID filter
        event_type: Optional event type filter
        severity: Optional severity filter
        status_filter: Optional status filter
        limit: Maximum number of results
        skip: Number of results to skip
        current_user: Authenticated user
        
    Returns:
        List of SafetyEventResponse objects
    """
    try:
        with db_manager.session_scope() as session:
            query = session.query(SafetyEventDB)
            
            if rig_id:
                query = query.filter(SafetyEventDB.rig_id == rig_id)
            if event_type:
                query = query.filter(SafetyEventDB.event_type == event_type)
            if severity:
                query = query.filter(SafetyEventDB.severity == severity)
            if status_filter:
                query = query.filter(SafetyEventDB.status == status_filter)
            
            # Order by most recent first
            query = query.order_by(SafetyEventDB.timestamp.desc())
            
            total = query.count()
            events = query.offset(skip).limit(limit).all()
            
            return [
                SafetyEventResponse(
                    id=event.id,
                    rig_id=event.rig_id,
                    event_type=event.event_type,
                    severity=event.severity,
                    status=event.status,
                    timestamp=event.timestamp,
                    resolved_at=event.resolved_at,
                    acknowledged_at=event.acknowledged_at,
                    reason=event.reason,
                    description=event.description,
                    recommendations=event.recommendations,
                    actions_taken=event.actions_taken
                )
                for event in events
            ]
            
    except Exception as e:
        logger.error(f"Error getting safety events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get safety events: {str(e)}"
        )


@router.post("/events/{event_id}/acknowledge")
async def acknowledge_safety_event(
    event_id: int,
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    Acknowledge a safety event
    
    Marks a safety event as acknowledged by an engineer.
    Requires engineer role or higher.
    
    Args:
        event_id: Safety event ID
        current_user: Authenticated user (must be engineer or higher)
        
    Returns:
        Success message
    """
    try:
        with db_manager.session_scope() as session:
            event = session.query(SafetyEventDB).filter(SafetyEventDB.id == event_id).first()
            
            if not event:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Safety event not found"
                )
            
            event.acknowledged_at = datetime.now()
            event.acknowledged_by = current_user.id
            session.commit()
            
            logger.info(f"Safety event {event_id} acknowledged by {current_user.username}")
            
            return {
                "success": True,
                "message": "Safety event acknowledged",
                "event_id": event_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging safety event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge safety event: {str(e)}"
        )


@router.post("/events/{event_id}/resolve")
async def resolve_safety_event(
    event_id: int,
    resolution_notes: Optional[str] = None,
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    Resolve a safety event
    
    Marks a safety event as resolved.
    Requires engineer role or higher.
    
    Args:
        event_id: Safety event ID
        resolution_notes: Optional notes about resolution
        current_user: Authenticated user (must be engineer or higher)
        
    Returns:
        Success message
    """
    try:
        with db_manager.session_scope() as session:
            event = session.query(SafetyEventDB).filter(SafetyEventDB.id == event_id).first()
            
            if not event:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Safety event not found"
                )
            
            event.status = "resolved"
            event.resolved_at = datetime.now()
            event.resolved_by = current_user.id
            if resolution_notes:
                if event.description:
                    event.description += f"\n\nResolution: {resolution_notes}"
                else:
                    event.description = f"Resolution: {resolution_notes}"
            
            session.commit()
            
            logger.info(f"Safety event {event_id} resolved by {current_user.username}")
            
            return {
                "success": True,
                "message": "Safety event resolved",
                "event_id": event_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving safety event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve safety event: {str(e)}"
        )

