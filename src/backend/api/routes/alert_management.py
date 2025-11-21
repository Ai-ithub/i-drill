"""
Alert Management API Routes
For managing alerts, acknowledgments, history, and escalation
"""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import logging

from services.alert_management_service import (
    alert_management_service,
    AlertSeverity,
    AlertType,
    AlertStatus
)
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert Management"])


# Request Models
class CreateAlertRequest(BaseModel):
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    rig_id: Optional[str] = Field(None, description="Rig ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    requires_acknowledgment: bool = Field(True, description="Whether alert requires acknowledgment")
    sound_alert: Optional[bool] = Field(None, description="Whether to play sound (None = auto-detect)")


class AcknowledgeAlertRequest(BaseModel):
    notes: Optional[str] = Field(None, description="Acknowledgment notes")


class ResolveAlertRequest(BaseModel):
    notes: Optional[str] = Field(None, description="Resolution notes")


@router.post("/create", response_model=Dict[str, Any])
async def create_alert(
    request: CreateAlertRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Create a new alert."""
    try:
        alert = alert_management_service.create_alert(
            alert_type=request.alert_type,
            severity=request.severity,
            title=request.title,
            message=request.message,
            rig_id=request.rig_id,
            metadata=request.metadata,
            requires_acknowledgment=request.requires_acknowledgment,
            sound_alert=request.sound_alert
        )
        return alert
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_alerts(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    current_user: User = Depends(get_current_user)
):
    """Get active alerts with optional filtering."""
    try:
        alerts = alert_management_service.get_active_alerts(
            rig_id=rig_id,
            severity=severity,
            alert_type=alert_type
        )
        return alerts
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_alert_history(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    current_user: User = Depends(get_current_user)
):
    """Get alert history with optional filtering."""
    try:
        alerts = alert_management_service.get_alert_history(
            rig_id=rig_id,
            severity=severity,
            alert_type=alert_type,
            status=status,
            limit=limit
        )
        return alerts
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(
    alert_id: int,
    request: AcknowledgeAlertRequest,
    current_user: User = Depends(get_current_user)
):
    """Acknowledge an alert."""
    try:
        alert = alert_management_service.acknowledge_alert(
            alert_id=alert_id,
            user_id=current_user.id,
            user_name=current_user.username,
            notes=request.notes
        )
        return alert
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: int,
    request: ResolveAlertRequest,
    current_user: User = Depends(get_current_user)
):
    """Resolve an alert."""
    try:
        alert = alert_management_service.resolve_alert(
            alert_id=alert_id,
            user_id=current_user.id,
            user_name=current_user.username,
            notes=request.notes
        )
        return alert
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{alert_id}", response_model=Dict[str, Any])
async def get_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific alert by ID."""
    try:
        # Check active alerts first
        active_alerts = alert_management_service.get_active_alerts()
        for alert in active_alerts:
            if alert["id"] == alert_id:
                return alert
        
        # Check history
        history = alert_management_service.get_alert_history(limit=10000)
        for alert in history:
            if alert["id"] == alert_id:
                return alert
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/escalation/check", response_model=List[Dict[str, Any]])
async def check_escalations(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Manually check for alerts that need escalation."""
    try:
        escalated = alert_management_service.check_escalations()
        return escalated
    except Exception as e:
        logger.error(f"Error checking escalations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_alert_stats(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    current_user: User = Depends(get_current_user)
):
    """Get alert statistics."""
    try:
        active_alerts = alert_management_service.get_active_alerts(rig_id=rig_id)
        history = alert_management_service.get_alert_history(rig_id=rig_id, limit=1000)
        
        # Count by severity
        severity_counts = {}
        for alert in active_alerts:
            severity = alert.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by status
        status_counts = {}
        for alert in active_alerts:
            status = alert.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count unacknowledged
        unacknowledged = sum(1 for a in active_alerts if a.get("status") == AlertStatus.ACTIVE.value)
        
        return {
            "total_active": len(active_alerts),
            "total_history": len(history),
            "unacknowledged": unacknowledged,
            "severity_counts": severity_counts,
            "status_counts": status_counts
        }
    except Exception as e:
        logger.error(f"Error getting alert stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

