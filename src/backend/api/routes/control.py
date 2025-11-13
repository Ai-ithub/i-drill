"""
Control API Routes
Handles applying changes to drilling parameters, change approval/rejection, and change history
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from datetime import datetime
import logging

from api.models.schemas import UserRole
from api.models.database_models import UserDB, ChangeRequestDB
from api.dependencies import get_current_active_user, get_current_engineer_user
from database import db_manager
import enum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/control", tags=["control"])


# Enums for Change Tracking
class ChangeStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class ChangeType(str, enum.Enum):
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    VALIDATION = "validation"


# Request/Response Schemas
from pydantic import BaseModel, Field
from typing import Dict, Any


class ChangeRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    change_type: ChangeType = Field(..., description="Type of change")
    component: str = Field(..., description="Component name")
    parameter: str = Field(..., description="Parameter name")
    value: Any = Field(..., description="New value for the parameter")
    auto_execute: bool = Field(False, description="Auto-execute if user has permission")
    metadata: Optional[Dict[str, Any]] = None


class ChangeResponse(BaseModel):
    success: bool
    change_id: int
    status: str
    message: str
    change_request: Optional[Dict[str, Any]] = None


class ChangeHistoryResponse(BaseModel):
    success: bool
    count: int
    changes: List[Dict[str, Any]]


@router.post("/apply-change", response_model=ChangeResponse)
async def apply_change(
    change_request: ChangeRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> ChangeResponse:
    """
    Apply a change to drilling parameters.
    
    Creates a change request that can be approved/rejected or auto-executed.
    If auto_execute is True and the user has admin/engineer role, the change
    will be applied immediately. Otherwise, it will be pending approval.
    
    Args:
        change_request: Change request details including rig_id, component, parameter, and value
        current_user: Authenticated user making the request
        
    Returns:
        ChangeResponse with success status, change_id, and status
        
    Raises:
        HTTPException: If change request creation fails
        
    Example:
        ```python
        {
            "rig_id": "RIG_01",
            "change_type": "optimization",
            "component": "drilling",
            "parameter": "rpm",
            "value": 120.0,
            "auto_execute": false
        }
        ```
    """
    try:
        with db_manager.session_scope() as session:
            # Determine if change should be auto-executed
            should_auto_execute = change_request.auto_execute and (
                current_user.role == UserRole.ADMIN.value or
                current_user.role == UserRole.ENGINEER.value
            )
            
            # Get current value if possible (could be fetched from sensor data or config)
            old_value = None
            # TODO: Fetch current value from sensor data or configuration
            
            # Determine status
            initial_status = "applied" if should_auto_execute else "pending"
            
            # Create change request
            change = ChangeRequestDB(
                rig_id=change_request.rig_id,
                change_type=change_request.change_type.value,
                component=change_request.component,
                parameter=change_request.parameter,
                old_value=old_value,
                new_value=str(change_request.value),
                status=initial_status,
                auto_execute=change_request.auto_execute,
                requested_by=current_user.id,
                approved_by=current_user.id if should_auto_execute else None,
                applied_by=current_user.id if should_auto_execute else None,
                approved_at=datetime.now() if should_auto_execute else None,
                applied_at=datetime.now() if should_auto_execute else None,
                metadata=change_request.metadata
            )
            
            session.add(change)
            session.commit()
            session.refresh(change)
            
            # If auto-execute, actually apply the change
            if should_auto_execute:
                try:
                    # Here you would integrate with actual control system
                    # For now, we just log it
                    logger.info(
                        f"Auto-executing change: {change_request.component}.{change_request.parameter} = {change_request.value} "
                        f"for rig {change_request.rig_id}"
                    )
                    # TODO: Integrate with actual drilling control system
                    # Example: control_system.set_parameter(rig_id, component, parameter, value)
                except Exception as e:
                    change.status = ChangeStatus.FAILED.value
                    change.error_message = str(e)
                    session.commit()
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to apply change: {str(e)}"
                    )
            
            logger.info(
                f"Change request created: ID={change.id}, Type={change_request.change_type.value}, "
                f"Status={change.status}, User={current_user.username}"
            )
            
            return ChangeResponse(
                success=True,
                change_id=change.id,
                status=change.status,
                message="Change applied successfully" if should_auto_execute else "Change request created and pending approval",
                change_request={
                    "id": change.id,
                    "rig_id": change.rig_id,
                    "change_type": change.change_type,
                    "component": change.component,
                    "parameter": change.parameter,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "status": change.status,
                    "requested_by": current_user.username,
                    "requested_at": change.requested_at.isoformat() if change.requested_at else None,
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply change: {str(e)}"
        )


@router.get("/change-history", response_model=ChangeHistoryResponse)
async def get_change_history(
    rig_id: Optional[str] = Query(None, description="Filter by rig ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get change history
    
    Returns a list of change requests with optional filtering.
    """
    try:
        with db_manager.session_scope() as session:
            query = session.query(ChangeRequestDB)
            
            if rig_id:
                query = query.filter(ChangeRequestDB.rig_id == rig_id)
            
            if status:
                query = query.filter(ChangeRequestDB.status == status)
            
            # Order by most recent first
            query = query.order_by(ChangeRequestDB.requested_at.desc())
            
            total = query.count()
            changes = query.offset(skip).limit(limit).all()
            
            changes_list = []
            for change in changes:
                changes_list.append({
                    "id": change.id,
                    "rig_id": change.rig_id,
                    "change_type": change.change_type,
                    "component": change.component,
                    "parameter": change.parameter,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "status": change.status,
                    "auto_execute": change.auto_execute,
                    "requested_at": change.requested_at.isoformat() if change.requested_at else None,
                    "approved_at": change.approved_at.isoformat() if change.approved_at else None,
                    "applied_at": change.applied_at.isoformat() if change.applied_at else None,
                    "rejection_reason": change.rejection_reason,
                    "error_message": change.error_message,
                    "metadata": change.metadata,
                })
            
            return ChangeHistoryResponse(
                success=True,
                count=total,
                changes=changes_list
            )
            
    except Exception as e:
        logger.error(f"Error getting change history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get change history: {str(e)}"
        )


@router.post("/change/{change_id}/approve")
async def approve_change(
    change_id: int,
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    Approve a pending change request
    
    Requires engineer role or higher.
    """
    try:
        with db_manager.session_scope() as session:
            change = session.query(ChangeRequestDB).filter(ChangeRequestDB.id == change_id).first()
            
            if not change:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Change request not found"
                )
            
            if change.status != "pending":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Change request is already {change.status}"
                )
            
            # Approve the change
            change.status = "approved"
            change.approved_by = current_user.id
            change.approved_at = datetime.now()
            
            # If auto_execute is enabled, apply it
            if change.auto_execute:
                try:
                    # Apply the change
                    logger.info(
                        f"Applying approved change: {change.component}.{change.parameter} = {change.new_value} "
                        f"for rig {change.rig_id}"
                    )
                    # TODO: Integrate with actual control system
                    change.status = "applied"
                    change.applied_by = current_user.id
                    change.applied_at = datetime.now()
                except Exception as e:
                    change.status = "failed"
                    change.error_message = str(e)
                    session.commit()
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to apply change: {str(e)}"
                    )
            
            session.commit()
            
            logger.info(f"Change request {change_id} approved by {current_user.username}")
            
            return {
                "success": True,
                "message": "Change request approved successfully",
                "change_id": change_id,
                "status": change.status
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve change: {str(e)}"
        )


@router.post("/change/{change_id}/reject")
async def reject_change(
    change_id: int,
    reason: Optional[str] = None,
    current_user: UserDB = Depends(get_current_engineer_user)
):
    """
    Reject a pending change request
    
    Requires engineer role or higher.
    """
    try:
        with db_manager.session_scope() as session:
            change = session.query(ChangeRequestDB).filter(ChangeRequestDB.id == change_id).first()
            
            if not change:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Change request not found"
                )
            
            if change.status != "pending":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Change request is already {change.status}"
                )
            
            # Reject the change
            change.status = "rejected"
            change.approved_by = current_user.id
            change.approved_at = datetime.now()
            change.rejection_reason = reason or "Rejected by engineer"
            
            session.commit()
            
            logger.info(f"Change request {change_id} rejected by {current_user.username}: {reason}")
            
            return {
                "success": True,
                "message": "Change request rejected",
                "change_id": change_id,
                "status": change.status,
                "rejection_reason": change.rejection_reason
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject change: {str(e)}"
        )


@router.get("/change/{change_id}")
async def get_change(
    change_id: int,
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get details of a specific change request
    """
    try:
        with db_manager.session_scope() as session:
            change = session.query(ChangeRequestDB).filter(ChangeRequestDB.id == change_id).first()
            
            if not change:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Change request not found"
                )
            
            return {
                "success": True,
                "change": {
                    "id": change.id,
                    "rig_id": change.rig_id,
                    "change_type": change.change_type,
                    "component": change.component,
                    "parameter": change.parameter,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "status": change.status,
                    "auto_execute": change.auto_execute,
                    "requested_at": change.requested_at.isoformat() if change.requested_at else None,
                    "approved_at": change.approved_at.isoformat() if change.approved_at else None,
                    "applied_at": change.applied_at.isoformat() if change.applied_at else None,
                    "rejection_reason": change.rejection_reason,
                    "error_message": change.error_message,
                    "metadata": change.metadata,
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get change: {str(e)}"
        )

