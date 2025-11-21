"""
Database Management API Routes
For optimization, retention, and archive management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from services.database_optimization_service import database_optimization_service
from services.data_retention_service import data_retention_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["Database Management"])


# Request Models
class CompressionRequest(BaseModel):
    table_name: str = Field("sensor_data", description="Table name")
    compress_after: str = Field("30 days", description="Compress data older than this")


@router.post("/optimize/timescaledb/enable", response_model=Dict[str, Any])
async def enable_timescaledb(
    current_user: User = Depends(require_role(["admin"]))
):
    """Enable TimescaleDB extension."""
    try:
        result = database_optimization_service.enable_timescaledb()
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to enable TimescaleDB")
            )
    except Exception as e:
        logger.error(f"Error enabling TimescaleDB: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/optimize/timescaledb/convert", response_model=Dict[str, Any])
async def convert_to_hypertable(
    table_name: str = "sensor_data",
    time_column: str = "timestamp",
    current_user: User = Depends(require_role(["admin"]))
):
    """Convert table to TimescaleDB hypertable."""
    try:
        result = database_optimization_service.convert_to_hypertable(table_name, time_column)
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to convert to hypertable")
            )
    except Exception as e:
        logger.error(f"Error converting to hypertable: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/optimize/timescaledb/compression", response_model=Dict[str, Any])
async def enable_compression(
    request: CompressionRequest,
    current_user: User = Depends(require_role(["admin"]))
):
    """Enable compression for old data in TimescaleDB."""
    try:
        result = database_optimization_service.enable_compression(
            request.table_name,
            request.compress_after
        )
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to enable compression")
            )
    except Exception as e:
        logger.error(f"Error enabling compression: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimize/indexes/stats", response_model=Dict[str, Any])
async def get_index_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get index usage statistics."""
    try:
        stats = database_optimization_service.get_index_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting index statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/optimize/analyze", response_model=Dict[str, Any])
async def analyze_table(
    table_name: str = "sensor_data",
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Run ANALYZE on table to update statistics."""
    try:
        result = database_optimization_service.analyze_table(table_name)
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to analyze table")
            )
    except Exception as e:
        logger.error(f"Error analyzing table: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimize/size", response_model=Dict[str, Any])
async def get_table_size(
    table_name: str = "sensor_data",
    current_user: User = Depends(get_current_user)
):
    """Get table size information."""
    try:
        size_info = database_optimization_service.get_table_size(table_name)
        return size_info
    except Exception as e:
        logger.error(f"Error getting table size: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimize/pool/stats", response_model=Dict[str, Any])
async def get_pool_stats(
    current_user: User = Depends(get_current_user)
):
    """Get connection pool statistics."""
    try:
        stats = database_optimization_service.get_connection_pool_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting pool stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/retention/cleanup", response_model=Dict[str, Any])
async def cleanup_old_data(
    dry_run: bool = False,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Clean up old data based on retention policy."""
    try:
        result = data_retention_service.cleanup_old_data(dry_run=dry_run)
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Cleanup failed")
            )
    except Exception as e:
        logger.error(f"Error cleaning up data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/retention/stats", response_model=Dict[str, Any])
async def get_retention_stats(
    current_user: User = Depends(get_current_user)
):
    """Get data retention statistics."""
    try:
        stats = data_retention_service.get_retention_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting retention stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/retention/start", response_model=Dict[str, Any])
async def start_retention_service(
    current_user: User = Depends(require_role(["admin"]))
):
    """Start automatic data retention service."""
    try:
        data_retention_service.start()
        return {
            "success": True,
            "message": "Data retention service started"
        }
    except Exception as e:
        logger.error(f"Error starting retention service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/retention/stop", response_model=Dict[str, Any])
async def stop_retention_service(
    current_user: User = Depends(require_role(["admin"]))
):
    """Stop automatic data retention service."""
    try:
        data_retention_service.stop()
        return {
            "success": True,
            "message": "Data retention service stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping retention service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

