"""
Health and Status API Routes
"""
from fastapi import APIRouter, HTTPException
from api.models.schemas import HealthCheck, ServiceStatus
from datetime import datetime
import logging
from services.kafka_service import kafka_service
from database_manager import db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint
    
    Returns the overall health status of the API and its services.
    """
    try:
        # Check service health
        services = {
            "database": _check_database_health(),
            "kafka": _check_kafka_health(),
            "api": True
        }
        
        # Determine overall health
        overall_status = "healthy" if all(services.values()) else "degraded"
        
        return HealthCheck(
            status=overall_status,
            version="1.0.0",
            services=services,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheck(
            status="unhealthy",
            version="1.0.0",
            services={"api": False},
            timestamp=datetime.now()
        )


@router.get("/services", response_model=list)
async def get_service_status():
    """
    Get detailed status of all services
    
    Returns detailed health information for each service.
    """
    try:
        services = []
        
        # Database status
        db_healthy = _check_database_health()
        services.append({
            "service_name": "postgresql",
            "status": "healthy" if db_healthy else "unhealthy",
            "is_healthy": db_healthy,
            "uptime_seconds": 0,  # TODO: Track actual uptime
            "last_check": datetime.now(),
            "details": {
                "connection_pool_size": 10,
                "active_connections": 0  # TODO: Get actual connections
            }
        })
        
        # Kafka status
        kafka_healthy = _check_kafka_health()
        services.append({
            "service_name": "kafka",
            "status": "healthy" if kafka_healthy else "unhealthy",
            "is_healthy": kafka_healthy,
            "uptime_seconds": 0,  # TODO: Track actual uptime
            "last_check": datetime.now(),
            "details": {
                "producer_initialized": kafka_service.producer is not None,
                "active_consumers": len(kafka_service.consumers)
            }
        })
        
        # API status
        services.append({
            "service_name": "api",
            "status": "healthy",
            "is_healthy": True,
            "uptime_seconds": 0,  # TODO: Track actual uptime
            "last_check": datetime.now(),
            "details": {
                "version": "1.0.0",
                "endpoints_count": 20  # Approximate count
            }
        })
        
        return services
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint
    
    Indicates whether the API is ready to accept traffic.
    """
    try:
        # Check critical services
        db_ready = _check_database_health()
        kafka_ready = _check_kafka_health()
        
        if not db_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        return {
            "status": "ready",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint
    
    Indicates whether the API is alive and running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now()
    }


def _check_database_health() -> bool:
    """Check database health"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def _check_kafka_health() -> bool:
    """Check Kafka health"""
    try:
        return kafka_service.producer is not None
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        return False

