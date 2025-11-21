"""
Drilling Sessions API Routes
Handles drilling session management (start, end, metrics tracking)
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from datetime import datetime
import logging

from api.models.database_models import UserDB, DrillingSessionDB
from api.dependencies import get_current_active_user, get_current_engineer_user
from database import db_manager
from services.data_service import DataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drilling-sessions", tags=["drilling-sessions"])

data_service = DataService()


@router.post("/start")
async def start_drilling_session(
    rig_id: str = Query(..., description="Rig identifier"),
    well_id: str = Query(..., description="Well identifier"),
    target_depth: float = Query(..., description="Target depth in feet"),
    start_depth: float = Query(0.0, description="Starting depth in feet"),
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    Start a new drilling session
    
    Creates a new drilling session record and begins tracking metrics.
    
    Args:
        rig_id: Rig identifier
        well_id: Well identifier
        target_depth: Target depth to reach
        start_depth: Starting depth (default: 0.0)
        current_user: Authenticated user (must be engineer or higher)
        
    Returns:
        Session information with session_id
    """
    try:
        # Check if there's an active session for this rig
        with db_manager.session_scope() as session:
            active_session = session.query(DrillingSessionDB).filter(
                DrillingSessionDB.rig_id == rig_id,
                DrillingSessionDB.status == "active"
            ).first()
            
            if active_session:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Active drilling session already exists for rig {rig_id} (session_id: {active_session.id})"
                )
            
            # Create new session
            new_session = DrillingSessionDB(
                rig_id=rig_id,
                well_id=well_id,
                start_time=datetime.now(),
                start_depth=start_depth,
                target_depth=target_depth,
                status="active"
            )
            session.add(new_session)
            session.commit()
            session.refresh(new_session)
            
            logger.info(f"Drilling session started: ID={new_session.id}, rig={rig_id}, well={well_id}, target_depth={target_depth}")
            
            return {
                "success": True,
                "message": "Drilling session started successfully",
                "session": {
                    "id": new_session.id,
                    "rig_id": new_session.rig_id,
                    "well_id": new_session.well_id,
                    "start_time": new_session.start_time.isoformat(),
                    "start_depth": new_session.start_depth,
                    "target_depth": target_depth,
                    "status": new_session.status
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting drilling session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start drilling session: {str(e)}"
        )


@router.post("/{session_id}/end")
async def end_drilling_session(
    session_id: int,
    end_reason: str = Query(..., description="Reason for ending session"),
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    End a drilling session
    
    Ends the drilling session and calculates final metrics.
    
    Args:
        session_id: Drilling session ID
        end_reason: Reason for ending the session
        current_user: Authenticated user (must be engineer or higher)
        
    Returns:
        Final session metrics
    """
    try:
        with db_manager.session_scope() as session:
            drilling_session = session.query(DrillingSessionDB).filter(
                DrillingSessionDB.id == session_id
            ).first()
            
            if not drilling_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Drilling session not found"
                )
            
            if drilling_session.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Session is already {drilling_session.status}"
                )
            
            # Get current depth from latest sensor data
            latest_data = data_service.get_latest_sensor_data(
                rig_id=drilling_session.rig_id,
                limit=1
            )
            end_depth = latest_data[0].get('depth', drilling_session.start_depth) if latest_data else drilling_session.start_depth
            
            # Calculate metrics
            end_time = datetime.now()
            total_time = (end_time - drilling_session.start_time).total_seconds() / 3600.0  # hours
            depth_drilled = end_depth - drilling_session.start_depth
            
            # Calculate average ROP
            if total_time > 0 and depth_drilled > 0:
                average_rop = depth_drilled / total_time
            else:
                average_rop = 0.0
            
            # Update session
            drilling_session.end_time = end_time
            drilling_session.end_depth = end_depth
            drilling_session.total_drilling_time_hours = total_time
            drilling_session.average_rop = average_rop
            drilling_session.status = "completed"
            drilling_session.notes = f"Ended: {end_reason or 'Session completed'}"
            
            session.commit()
            
            logger.info(f"Drilling session ended: ID={session_id}, depth_drilled={depth_drilled:.2f}ft, time={total_time:.2f}hrs")
            
            return {
                "success": True,
                "message": "Drilling session ended successfully",
                "session": {
                    "id": drilling_session.id,
                    "rig_id": drilling_session.rig_id,
                    "well_id": drilling_session.well_id,
                    "start_time": drilling_session.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "start_depth": drilling_session.start_depth,
                    "end_depth": end_depth,
                    "depth_drilled": depth_drilled,
                    "total_drilling_time_hours": total_time,
                    "average_rop": average_rop,
                    "status": drilling_session.status
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending drilling session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end drilling session: {str(e)}"
        )


@router.get("/")
async def get_drilling_sessions(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    well_id: Optional[str] = Query(None, description="Filter by well ID"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get drilling sessions with optional filtering
    
    Returns a list of drilling sessions with optional filtering.
    
    Args:
        rig_id: Optional rig ID filter
        well_id: Optional well ID filter
        status_filter: Optional status filter
        limit: Maximum number of results
        skip: Number of results to skip
        current_user: Authenticated user
        
    Returns:
        List of drilling sessions
    """
    try:
        with db_manager.session_scope() as session:
            query = session.query(DrillingSessionDB)
            
            if rig_id:
                query = query.filter(DrillingSessionDB.rig_id == rig_id)
            if well_id:
                query = query.filter(DrillingSessionDB.well_id == well_id)
            if status_filter:
                query = query.filter(DrillingSessionDB.status == status_filter)
            
            # Order by most recent first
            query = query.order_by(DrillingSessionDB.start_time.desc())
            
            total = query.count()
            sessions = query.offset(skip).limit(limit).all()
            
            return {
                "success": True,
                "count": total,
                "sessions": [
                    {
                        "id": s.id,
                        "rig_id": s.rig_id,
                        "well_id": s.well_id,
                        "start_time": s.start_time.isoformat(),
                        "end_time": s.end_time.isoformat() if s.end_time else None,
                        "start_depth": s.start_depth,
                        "end_depth": s.end_depth,
                        "average_rop": s.average_rop,
                        "total_drilling_time_hours": s.total_drilling_time_hours,
                        "status": s.status,
                        "notes": s.notes
                    }
                    for s in sessions
                ]
            }
            
    except Exception as e:
        logger.error(f"Error getting drilling sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drilling sessions: {str(e)}"
        )


@router.get("/{session_id}")
async def get_drilling_session(
    session_id: int,
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get details of a specific drilling session
    
    Args:
        session_id: Drilling session ID
        current_user: Authenticated user
        
    Returns:
        Session details
    """
    try:
        with db_manager.session_scope() as session:
            drilling_session = session.query(DrillingSessionDB).filter(
                DrillingSessionDB.id == session_id
            ).first()
            
            if not drilling_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Drilling session not found"
                )
            
            return {
                "success": True,
                "session": {
                    "id": drilling_session.id,
                    "rig_id": drilling_session.rig_id,
                    "well_id": drilling_session.well_id,
                    "start_time": drilling_session.start_time.isoformat(),
                    "end_time": drilling_session.end_time.isoformat() if drilling_session.end_time else None,
                    "start_depth": drilling_session.start_depth,
                    "end_depth": drilling_session.end_depth,
                    "average_rop": drilling_session.average_rop,
                    "total_drilling_time_hours": drilling_session.total_drilling_time_hours,
                    "status": drilling_session.status,
                    "notes": drilling_session.notes
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting drilling session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drilling session: {str(e)}"
        )

