"""
Maintenance API Routes
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from api.models.schemas import (
    MaintenanceAlert,
    MaintenanceSchedule,
    CreateMaintenanceAlertRequest,
    UpdateMaintenanceScheduleRequest,
    MaintenanceAlertAcknowledgeRequest,
    MaintenanceAlertResolveRequest,
)
from datetime import datetime, timedelta
import logging

from services.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maintenance", tags=["maintenance"])
data_service = DataService()


@router.get("/alerts", response_model=List[MaintenanceAlert])
async def get_maintenance_alerts(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$", description="Filter by severity"),
    hours: int = Query(24, ge=1, le=720, description="Hours of history to retrieve")
):
    """
    Get maintenance alerts
    
    Returns recent maintenance alerts for drilling equipment.
    """
    try:
        alerts = data_service.get_maintenance_alerts(
            rig_id=rig_id,
            severity=severity,
            resolved=None,
            limit=100,
            hours=hours
        )
        return [MaintenanceAlert(**alert) for alert in alerts]
        
    except Exception as e:
        logger.error(f"Error getting maintenance alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}", response_model=MaintenanceAlert)
async def get_maintenance_alert(alert_id: str):
    """
    Get a specific maintenance alert
    """
    try:
        alert = data_service.get_maintenance_alert_by_id(int(alert_id))
        if not alert:
            raise HTTPException(status_code=404, detail="Maintenance alert not found")
        
        return MaintenanceAlert(**alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting maintenance alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts", response_model=MaintenanceAlert, status_code=201)
async def create_maintenance_alert(request: CreateMaintenanceAlertRequest):
    """
    Create a maintenance alert entry
    """
    try:
        alert_id = data_service.create_maintenance_alert(request.model_dump())
        if alert_id is None:
            raise HTTPException(status_code=500, detail="Failed to create maintenance alert")
        
        alert = data_service.get_maintenance_alert_by_id(alert_id)
        if not alert:
            raise HTTPException(status_code=500, detail="Maintenance alert creation verification failed")
        
        return MaintenanceAlert(**alert)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating maintenance alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge", response_model=MaintenanceAlert)
async def acknowledge_maintenance_alert(alert_id: str, request: MaintenanceAlertAcknowledgeRequest):
    try:
        alert = data_service.acknowledge_maintenance_alert(
            int(alert_id),
            acknowledged_by=request.acknowledged_by,
            notes=request.notes,
            dvr_history_id=request.dvr_history_id,
        )
        if alert is None:
            raise HTTPException(status_code=404, detail="Maintenance alert not found")
        return MaintenanceAlert(**alert)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging maintenance alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve", response_model=MaintenanceAlert)
async def resolve_maintenance_alert(alert_id: str, request: MaintenanceAlertResolveRequest):
    try:
        alert = data_service.resolve_maintenance_alert(
            int(alert_id),
            resolved_by=request.resolved_by,
            notes=request.notes,
            dvr_history_id=request.dvr_history_id,
        )
        if alert is None:
            raise HTTPException(status_code=404, detail="Maintenance alert not found")
        return MaintenanceAlert(**alert)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving maintenance alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule", response_model=List[MaintenanceSchedule])
async def get_maintenance_schedule(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    status: Optional[str] = Query(None, pattern="^(scheduled|in_progress|completed|cancelled)$"),
    days_ahead: int = Query(30, ge=1, le=365, description="Days ahead to schedule")
):
    """
    Get maintenance schedule
    
    Returns scheduled maintenance activities.
    """
    try:
        schedules = data_service.get_maintenance_schedules(
            rig_id=rig_id,
            status=status,
            limit=100,
            until_date=datetime.now() + timedelta(days=days_ahead)
        )
        return [MaintenanceSchedule(**schedule) for schedule in schedules]
        
    except Exception as e:
        logger.error(f"Error getting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule", response_model=MaintenanceSchedule, status_code=201)
async def create_maintenance_schedule(schedule: MaintenanceSchedule):
    """
    Create a new maintenance schedule
    
    Adds a new scheduled maintenance activity.
    """
    try:
        payload = schedule.model_dump(
            exclude={"id", "created_at", "updated_at"},
            exclude_unset=True,
        )
        created_schedule = data_service.create_maintenance_schedule(payload)
        
        if created_schedule is None:
            raise HTTPException(status_code=500, detail="Failed to create maintenance schedule")
        
        return MaintenanceSchedule(**created_schedule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedule/{schedule_id}", response_model=MaintenanceSchedule)
async def update_maintenance_schedule(schedule_id: str, schedule: UpdateMaintenanceScheduleRequest):
    """
    Update an existing maintenance schedule
    """
    try:
        update_data = schedule.model_dump(exclude_unset=True)
        updated_schedule = data_service.update_maintenance_schedule(int(schedule_id), update_data)
        
        if updated_schedule is None:
            raise HTTPException(status_code=404, detail="Maintenance schedule not found")
        
        return MaintenanceSchedule(**updated_schedule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedule/{schedule_id}")
async def delete_maintenance_schedule(schedule_id: str):
    """
    Delete a maintenance schedule
    """
    try:
        deleted = data_service.delete_maintenance_schedule(int(schedule_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Maintenance schedule not found")
        
        return {
            "success": True,
            "message": "Maintenance schedule deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

