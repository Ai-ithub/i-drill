"""
Configuration API Routes
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from api.models.schemas import (
    WellProfileConfig,
    DrillingParametersConfig,
    ErrorResponse
)
from api.models.database_models import WellProfileDB
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("/well-profiles")
async def get_well_profiles(
    rig_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get well profile configurations
    
    Args:
        rig_id: Optional filter by rig ID
        
    Returns:
        List of well profiles
    """
    try:
        query = db.query(WellProfileDB)
        
        if rig_id:
            query = query.filter(WellProfileDB.rig_id == rig_id)
        
        profiles = query.all()
        
        return {
            "success": True,
            "count": len(profiles),
            "profiles": [
                {
                    "id": p.id,
                    "well_id": p.well_id,
                    "rig_id": p.rig_id,
                    "total_depth": p.total_depth,
                    "kick_off_point": p.kick_off_point,
                    "build_rate": p.build_rate,
                    "max_inclination": p.max_inclination,
                    "target_zone_start": p.target_zone_start,
                    "target_zone_end": p.target_zone_end,
                    "geological_data": p.geological_data,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat()
                }
                for p in profiles
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting well profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/well-profiles/{well_id}")
async def get_well_profile(
    well_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific well profile
    
    Args:
        well_id: Well identifier
        
    Returns:
        Well profile details
    """
    try:
        profile = db.query(WellProfileDB).filter(
            WellProfileDB.well_id == well_id
        ).first()
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Well profile {well_id} not found"
            )
        
        return {
            "success": True,
            "profile": {
                "id": profile.id,
                "well_id": profile.well_id,
                "rig_id": profile.rig_id,
                "total_depth": profile.total_depth,
                "kick_off_point": profile.kick_off_point,
                "build_rate": profile.build_rate,
                "max_inclination": profile.max_inclination,
                "target_zone_start": profile.target_zone_start,
                "target_zone_end": profile.target_zone_end,
                "geological_data": profile.geological_data,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting well profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/well-profiles")
async def create_well_profile(
    config: WellProfileConfig,
    db: Session = Depends(get_db)
):
    """
    Create a new well profile
    
    Args:
        config: Well profile configuration
        
    Returns:
        Created well profile
    """
    try:
        # Check if well_id already exists
        existing = db.query(WellProfileDB).filter(
            WellProfileDB.well_id == config.well_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Well profile {config.well_id} already exists"
            )
        
        # Create new profile
        profile = WellProfileDB(
            well_id=config.well_id,
            rig_id="RIG_01",  # Default or get from config
            total_depth=config.total_depth,
            kick_off_point=config.kick_off_point,
            build_rate=config.build_rate,
            max_inclination=config.max_inclination,
            target_zone_start=config.target_zone_start,
            target_zone_end=config.target_zone_end
        )
        
        db.add(profile)
        db.commit()
        db.refresh(profile)
        
        return {
            "success": True,
            "message": "Well profile created successfully",
            "profile": {
                "id": profile.id,
                "well_id": profile.well_id,
                "rig_id": profile.rig_id,
                "total_depth": profile.total_depth
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating well profile: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/well-profiles/{well_id}")
async def update_well_profile(
    well_id: str,
    config: WellProfileConfig,
    db: Session = Depends(get_db)
):
    """
    Update an existing well profile
    
    Args:
        well_id: Well identifier
        config: Updated well profile configuration
        
    Returns:
        Updated well profile
    """
    try:
        profile = db.query(WellProfileDB).filter(
            WellProfileDB.well_id == well_id
        ).first()
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Well profile {well_id} not found"
            )
        
        # Update fields
        profile.total_depth = config.total_depth
        profile.kick_off_point = config.kick_off_point
        profile.build_rate = config.build_rate
        profile.max_inclination = config.max_inclination
        profile.target_zone_start = config.target_zone_start
        profile.target_zone_end = config.target_zone_end
        
        db.commit()
        db.refresh(profile)
        
        return {
            "success": True,
            "message": "Well profile updated successfully",
            "profile": {
                "id": profile.id,
                "well_id": profile.well_id,
                "updated_at": profile.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating well profile: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/well-profiles/{well_id}")
async def delete_well_profile(
    well_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a well profile
    
    Args:
        well_id: Well identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        profile = db.query(WellProfileDB).filter(
            WellProfileDB.well_id == well_id
        ).first()
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Well profile {well_id} not found"
            )
        
        db.delete(profile)
        db.commit()
        
        return {
            "success": True,
            "message": f"Well profile {well_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting well profile: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_config():
    """
    Get system configuration
    
    Returns:
        System configuration settings
    """
    return {
        "success": True,
        "config": {
            "api_version": "1.0.0",
            "max_upload_size_mb": 100,
            "data_retention_days": 365,
            "default_rig_id": "RIG_01",
            "supported_models": ["lstm", "transformer", "cnn_lstm"],
            "real_time_update_interval_ms": 1000,
            "max_concurrent_connections": 100
        }
    }
