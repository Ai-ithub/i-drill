"""
Performance Metrics API Routes
Handles real-time performance metrics calculation
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional
import logging

from api.models.database_models import UserDB
from api.dependencies import get_current_active_user
from services.performance_metrics_service import performance_metrics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/metrics/{rig_id}")
async def get_performance_metrics(
    rig_id: str,
    session_id: Optional[int] = Query(None, description="Optional drilling session ID"),
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get real-time performance metrics for a rig
    
    Calculates and returns:
    - ROP Efficiency
    - Energy Efficiency
    - Bit Life Remaining
    - Drilling Efficiency Index (DEI)
    - Cost per Meter (if session_id provided)
    - Total Cost (if session_id provided)
    - Projected Total Cost (if session_id provided)
    - Estimated Time to Target (if session_id and target_depth provided)
    
    Args:
        rig_id: Rig identifier
        session_id: Optional drilling session ID for cost calculations
        current_user: Authenticated user
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        metrics = performance_metrics_service.calculate_real_time_metrics(
            rig_id=rig_id,
            session_id=session_id
        )
        
        if "error" in metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=metrics["error"]
            )
        
        return {
            "success": True,
            "rig_id": rig_id,
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate performance metrics: {str(e)}"
        )

