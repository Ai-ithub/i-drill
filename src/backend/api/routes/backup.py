"""
Backup API Routes
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from datetime import datetime
import logging

from services.backup_service import backup_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backup", tags=["Backup"])


@router.post("/create", summary="Create manual backup")
async def create_backup(
    include_database: bool = True,
    include_models: bool = True,
    include_config: bool = True,
    include_logs: bool = False
) -> Dict[str, Any]:
    """
    Create a manual backup of the system
    
    - **include_database**: Include database backup
    - **include_models**: Include ML models
    - **include_config**: Include configuration files
    - **include_logs**: Include log files
    """
    try:
        result = backup_service.create_backup(
            include_database=include_database,
            include_models=include_models,
            include_config=include_config,
            include_logs=include_logs
        )
        return {
            "success": True,
            "message": "Backup created successfully",
            "backup": result
        }
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backup creation failed: {str(e)}"
        )


@router.get("/list", summary="List all backups")
async def list_backups() -> Dict[str, Any]:
    """List all available backups"""
    try:
        backups = backup_service.list_backups()
        return {
            "success": True,
            "count": len(backups),
            "backups": backups
        }
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list backups: {str(e)}"
        )


@router.post("/restore", summary="Restore from backup")
async def restore_backup(backup_path: str) -> Dict[str, Any]:
    """
    Restore system from a backup
    
    - **backup_path**: Path to backup file
    """
    try:
        result = backup_service.restore_backup(backup_path)
        return {
            "success": True,
            "message": "Backup restore initiated",
            "result": result
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Backup restore failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backup restore failed: {str(e)}"
        )


@router.get("/status", summary="Get backup service status")
async def get_backup_status() -> Dict[str, Any]:
    """Get backup service status and configuration"""
    return {
        "success": True,
        "enabled": backup_service.enabled,
        "backup_dir": str(backup_service.backup_dir),
        "retention_days": backup_service.retention_days,
        "scheduler_running": backup_service.scheduler.running if backup_service.scheduler else False
    }

