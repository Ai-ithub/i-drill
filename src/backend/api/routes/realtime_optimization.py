"""
Real-time Optimization API Routes
For RL model integration, performance metrics, cost tracking, and optimization recommendations
"""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from services.realtime_optimization_service import realtime_optimization_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization/realtime", tags=["Real-time Optimization"])


# Request Models
class LoadRLModelRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    model_path: str = Field(..., description="Path to trained model")
    model_type: str = Field("PPO", description="Model type (PPO or SAC)")
    auto_apply: bool = Field(False, description="Whether to auto-apply recommendations")


@router.post("/rl/load-model", response_model=Dict[str, Any])
async def load_rl_model(
    request: LoadRLModelRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Load RL model (PPO/SAC) for a rig."""
    try:
        result = realtime_optimization_service.load_rl_model(
            rig_id=request.rig_id,
            model_path=request.model_path,
            model_type=request.model_type,
            auto_apply=request.auto_apply
        )
        return result
    except Exception as e:
        logger.error(f"Error loading RL model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/rl/recommendation/{rig_id}", response_model=Dict[str, Any])
async def get_rl_recommendation(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time parameter recommendation from RL model."""
    try:
        result = realtime_optimization_service.get_realtime_recommendation(rig_id)
        return result
    except Exception as e:
        logger.error(f"Error getting RL recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/performance/{rig_id}", response_model=Dict[str, Any])
async def get_performance_metrics(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time performance metrics (ROP Efficiency, Energy Efficiency, DEI)."""
    try:
        result = realtime_optimization_service.calculate_realtime_performance_metrics(rig_id)
        return result
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cost/{rig_id}", response_model=Dict[str, Any])
async def get_cost_tracking(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time cost tracking (Cost per Meter, Total Cost, Budget comparison)."""
    try:
        result = realtime_optimization_service.calculate_realtime_cost(rig_id)
        return result
    except Exception as e:
        logger.error(f"Error calculating cost: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/recommendations/{rig_id}", response_model=Dict[str, Any])
async def get_optimization_recommendations(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get optimization recommendations (ROP improvement, cost reduction, energy reduction)."""
    try:
        result = realtime_optimization_service.generate_optimization_recommendations(rig_id)
        return result
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/summary/{rig_id}", response_model=Dict[str, Any])
async def get_optimization_summary(
    rig_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get complete optimization summary (metrics, cost, recommendations)."""
    try:
        # Get all data
        metrics_result = realtime_optimization_service.calculate_realtime_performance_metrics(rig_id)
        cost_result = realtime_optimization_service.calculate_realtime_cost(rig_id)
        recommendations_result = realtime_optimization_service.generate_optimization_recommendations(rig_id)
        rl_recommendation_result = realtime_optimization_service.get_realtime_recommendation(rig_id)
        
        return {
            "success": True,
            "rig_id": rig_id,
            "metrics": metrics_result.get("metrics") if metrics_result.get("success") else None,
            "cost": cost_result.get("cost") if cost_result.get("success") else None,
            "recommendations": recommendations_result.get("recommendations") if recommendations_result.get("success") else [],
            "rl_recommendation": rl_recommendation_result.get("recommendation") if rl_recommendation_result.get("success") else None,
            "timestamp": metrics_result.get("metrics", {}).get("timestamp") if metrics_result.get("success") else None
        }
    except Exception as e:
        logger.error(f"Error getting optimization summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_optimization_monitoring(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Start real-time optimization monitoring."""
    try:
        realtime_optimization_service.start_monitoring()
        return {
            "success": True,
            "message": "Optimization monitoring started"
        }
    except Exception as e:
        logger.error(f"Error starting optimization monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/stop", response_model=Dict[str, Any])
async def stop_optimization_monitoring(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Stop real-time optimization monitoring."""
    try:
        realtime_optimization_service.stop_monitoring()
        return {
            "success": True,
            "message": "Optimization monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping optimization monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/status", response_model=Dict[str, Any])
async def get_monitoring_status(
    current_user: User = Depends(get_current_user)
):
    """Get optimization monitoring status."""
    return {
        "running": realtime_optimization_service.running,
        "monitoring_interval": realtime_optimization_service.monitoring_interval,
        "loaded_models": list(realtime_optimization_service.rl_models.keys()),
        "auto_apply_enabled": realtime_optimization_service.auto_apply_enabled
    }

