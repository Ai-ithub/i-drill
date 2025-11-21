"""
Kafka Management API Routes
For optimization, monitoring, and topic management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from services.kafka_optimization_service import kafka_optimization_service
from services.kafka_monitoring_service import kafka_monitoring_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kafka", tags=["Kafka Management"])


# Request Models
class TopicCreationRequest(BaseModel):
    topic_name: str = Field(..., description="Topic name")
    num_partitions: int = Field(4, description="Number of partitions (minimum 4)")
    replication_factor: int = Field(1, description="Replication factor")
    retention_ms: int = Field(2592000000, description="Retention time in milliseconds (30 days default)")
    compression_type: str = Field("snappy", description="Compression type: none, gzip, snappy, lz4, zstd")
    segment_ms: int = Field(86400000, description="Segment roll time in milliseconds (1 day)")


class TopicConfigUpdateRequest(BaseModel):
    config_updates: Dict[str, str] = Field(..., description="Configuration key-value pairs to update")


@router.post("/topics/optimize", response_model=Dict[str, Any])
async def optimize_sensor_topic(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """
    Ensure sensor data topic is optimized with proper partitions and settings.
    Creates topic if it doesn't exist, or updates configuration if needed.
    """
    try:
        result = kafka_optimization_service.ensure_sensor_topic_optimized()
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to optimize topic")
            )
    except Exception as e:
        logger.error(f"Error optimizing topic: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/topics/create", response_model=Dict[str, Any])
async def create_optimized_topic(
    request: TopicCreationRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Create a new Kafka topic with optimized settings."""
    try:
        result = kafka_optimization_service.create_optimized_topic(
            topic_name=request.topic_name,
            num_partitions=request.num_partitions,
            replication_factor=request.replication_factor,
            retention_ms=request.retention_ms,
            compression_type=request.compression_type,
            segment_ms=request.segment_ms
        )
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to create topic")
            )
    except Exception as e:
        logger.error(f"Error creating topic: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/topics/{topic_name}/config", response_model=Dict[str, Any])
async def update_topic_config(
    topic_name: str,
    request: TopicConfigUpdateRequest,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Update topic configuration."""
    try:
        result = kafka_optimization_service.update_topic_config(
            topic_name=topic_name,
            config_updates=request.config_updates
        )
        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to update topic config")
            )
    except Exception as e:
        logger.error(f"Error updating topic config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/topics/{topic_name}/config", response_model=Dict[str, Any])
async def get_topic_config(
    topic_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get current topic configuration."""
    try:
        config = kafka_optimization_service.get_topic_config(topic_name)
        if config:
            return {"topic": topic_name, "config": config}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic {topic_name} not found or error retrieving config"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/metrics", response_model=Dict[str, Any])
async def get_kafka_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get Kafka performance metrics (throughput, lag, error rate)."""
    try:
        metrics = kafka_monitoring_service.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/throughput", response_model=Dict[str, Any])
async def get_throughput_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get throughput metrics (messages/sec, bytes/sec)."""
    try:
        throughput = kafka_monitoring_service.get_throughput()
        return throughput
    except Exception as e:
        logger.error(f"Error getting throughput: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/consumer-lag", response_model=Dict[str, Any])
async def get_consumer_lag(
    consumer_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get consumer lag metrics."""
    try:
        lag = kafka_monitoring_service.get_consumer_lag(consumer_id)
        return lag
    except Exception as e:
        logger.error(f"Error getting consumer lag: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/monitoring/error-rate", response_model=Dict[str, Any])
async def get_error_rate(
    current_user: User = Depends(get_current_user)
):
    """Get error rate metrics."""
    try:
        error_rate = kafka_monitoring_service.get_error_rate()
        return error_rate
    except Exception as e:
        logger.error(f"Error getting error rate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_monitoring(
    interval: float = 5.0,
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Start Kafka monitoring service."""
    try:
        kafka_monitoring_service.start_monitoring(interval=interval)
        return {
            "success": True,
            "message": "Kafka monitoring started",
            "interval": interval
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/stop", response_model=Dict[str, Any])
async def stop_monitoring(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Stop Kafka monitoring service."""
    try:
        kafka_monitoring_service.stop_monitoring()
        return {
            "success": True,
            "message": "Kafka monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monitoring/reset", response_model=Dict[str, Any])
async def reset_metrics(
    current_user: User = Depends(require_role(["admin", "engineer"]))
):
    """Reset monitoring metrics counters."""
    try:
        kafka_monitoring_service.reset_metrics()
        return {
            "success": True,
            "message": "Metrics reset"
        }
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

