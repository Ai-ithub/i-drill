"""
Drilling Events API Routes
Handles formation change detection and other drilling events
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from datetime import datetime
import logging

from api.models.schemas import (
    FormationChangeDetectionResponse,
    SensorDataPoint
)
from api.models.database_models import UserDB, DrillingEventDB
from api.dependencies import get_current_active_user
from database import db_manager
from services.drilling_events_service import drilling_events_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drilling-events", tags=["drilling-events"])


@router.post("/detect-formation-change", response_model=FormationChangeDetectionResponse)
async def detect_formation_change(
    sensor_data: SensorDataPoint,
    current_user: UserDB = Depends(get_current_active_user)
) -> FormationChangeDetectionResponse:
    """
    Detect formation change based on sensor data
    
    Analyzes current sensor data to detect potential formation changes:
    - Gamma ray changes
    - Resistivity changes
    - ROP pattern changes
    
    Args:
        sensor_data: Current sensor data point
        current_user: Authenticated user
        
    Returns:
        FormationChangeDetectionResponse with detection result and recommended parameters
    """
    try:
        result = drilling_events_service.detect_formation_change(sensor_data.model_dump())
        
        return FormationChangeDetectionResponse(
            formation_change_detected=result.get("formation_change_detected", False),
            depth=result.get("depth", 0.0),
            confidence=result.get("confidence", 0.0),
            formation_type=result.get("formation_type"),
            recommended_parameters=result.get("recommended_parameters", {}),
            event_id=result.get("event_id"),
            timestamp=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat()))
        )
        
    except Exception as e:
        logger.error(f"Error detecting formation change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect formation change: {str(e)}"
        )


@router.get("/events")
async def get_drilling_events(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get drilling events with optional filtering
    
    Returns a list of drilling events (formation changes, performance alerts, etc.)
    with optional filtering.
    
    Args:
        rig_id: Optional rig ID filter
        event_type: Optional event type filter
        severity: Optional severity filter
        limit: Maximum number of results
        skip: Number of results to skip
        current_user: Authenticated user
        
    Returns:
        List of drilling events
    """
    try:
        with db_manager.session_scope() as session:
            query = session.query(DrillingEventDB)
            
            if rig_id:
                query = query.filter(DrillingEventDB.rig_id == rig_id)
            if event_type:
                query = query.filter(DrillingEventDB.event_type == event_type)
            if severity:
                query = query.filter(DrillingEventDB.severity == severity)
            
            # Order by most recent first
            query = query.order_by(DrillingEventDB.timestamp.desc())
            
            total = query.count()
            events = query.offset(skip).limit(limit).all()
            
            return {
                "success": True,
                "count": total,
                "events": [
                    {
                        "id": event.id,
                        "rig_id": event.rig_id,
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "timestamp": event.timestamp.isoformat(),
                        "depth": event.depth,
                        "description": event.description,
                        "metadata": event.metadata,
                        "acknowledged": event.acknowledged,
                        "acknowledged_at": event.acknowledged_at.isoformat() if event.acknowledged_at else None
                    }
                    for event in events
                ]
            }
            
    except Exception as e:
        logger.error(f"Error getting drilling events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drilling events: {str(e)}"
        )

