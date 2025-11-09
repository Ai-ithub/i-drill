"""
Health and Status API Routes
"""
from fastapi import APIRouter, HTTPException
from api.models.schemas import HealthCheckResponse
from datetime import datetime
import logging
from services.kafka_service import kafka_service
from database import check_database_health, db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the overall health status of the API and its services.
    """
    try:
        # Check database health
        db_health = check_database_health()
        db_available = getattr(db_manager, "is_available", lambda: False)()
        db_healthy = db_health.get("database") == "healthy" and db_available

        # Check Kafka health
        kafka_available = kafka_service.is_available()
        kafka_healthy = kafka_available and _check_kafka_health()

        # Check service health
        services = {
            "database": db_healthy,
            "kafka": kafka_healthy,
            "api": True
        }

        # Determine overall health
        overall_status = "healthy" if all(services.values()) else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            version="1.0.0",
            services=services,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            services={"api": False},
            timestamp=datetime.now()
        )


@router.get("/database")
async def database_health():
    """
    Database-specific health check
    
    Returns detailed database health information.
    """
    try:
        health = check_database_health()
        is_healthy = health.get("database") == "healthy" and getattr(db_manager, "is_available", lambda: False)()
        
        return {
            "service": "database",
            "status": "healthy" if is_healthy else "unhealthy",
            "details": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return {
            "service": "database",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/kafka")
async def kafka_health():
    """
    Kafka-specific health check
    
    Returns Kafka service health information.
    """
    try:
        is_healthy = kafka_service.is_available() and _check_kafka_health()
        
        return {
            "service": "kafka",
            "status": "healthy" if is_healthy else "unhealthy",
            "details": {
                "producer_initialized": kafka_service.producer is not None,
                "library_installed": getattr(kafka_service, "available", False),
                "active_consumers": len(kafka_service.consumers)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Kafka health check error: {e}")
        return {
            "service": "kafka",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint
    
    Indicates whether the API is ready to accept traffic.
    Requires database to be healthy.
    """
    try:
        # Check critical services
        db_health = check_database_health()
        db_ready = db_health.get("database") == "healthy" and getattr(db_manager, "is_available", lambda: False)()
        
        if not db_ready:
            raise HTTPException(
                status_code=503, 
                detail="Database not ready"
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Service not ready"
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint
    
    Indicates whether the API is alive and running.
    Always returns 200 if the service is running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/services")
async def get_services_status():
    """
    Get detailed status of all services
    
    Returns detailed health information for each service.
    """
    try:
        services = []
        
        # Database status
        db_health = check_database_health()
        db_healthy = db_health.get("database") == "healthy" and getattr(db_manager, "is_available", lambda: False)()
        
        services.append({
            "name": "postgresql",
            "status": "healthy" if db_healthy else "unhealthy",
            "is_healthy": db_healthy,
            "last_check": datetime.now().isoformat(),
            "details": db_health
        })
        
        # Kafka status
        kafka_healthy = _check_kafka_health()
        services.append({
            "name": "kafka",
            "status": "healthy" if kafka_healthy else "unhealthy",
            "is_healthy": kafka_healthy,
            "last_check": datetime.now().isoformat(),
            "details": {
                "producer_initialized": kafka_service.producer is not None,
                "library_installed": getattr(kafka_service, "available", False),
                "active_consumers": len(kafka_service.consumers)
            }
        })
        
        # API status
        services.append({
            "name": "api",
            "status": "healthy",
            "is_healthy": True,
            "last_check": datetime.now().isoformat(),
            "details": {
                "version": "1.0.0"
            }
        })
        
        return {
            "services": services,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _check_kafka_health() -> bool:
    """
    Check Kafka health
    
    Returns:
        True if Kafka is healthy, False otherwise
    """
    try:
        if not kafka_service.is_available():
            return False
        return kafka_service.check_connection()
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        return False
