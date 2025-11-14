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
        # Database status
        db_health = check_database_health()
        db_healthy = db_health.get("database") == "healthy" and getattr(db_manager, "is_available", lambda: False)()
        db_status = "healthy" if db_healthy else "unhealthy"
        
        # Kafka status
        kafka_healthy = _check_kafka_health()
        kafka_status = "healthy" if kafka_healthy else "unhealthy"
        kafka_available = kafka_service.is_available()
        
        # RL Environment status (check if available)
        try:
            from services.rl_service import rl_service
            rl_status = "available" if hasattr(rl_service, 'is_available') and rl_service.is_available() else "unavailable"
            rl_message = "RL environment is available" if rl_status == "available" else "RL environment is not available"
        except Exception:
            rl_status = "unavailable"
            rl_message = "RL environment service not initialized"
        
        # MLflow status (check if available)
        try:
            from services.mlflow_service import mlflow_service
            mlflow_status = "available" if hasattr(mlflow_service, 'is_available') and mlflow_service.is_available() else "unavailable"
            mlflow_message = "MLflow is available" if mlflow_status == "available" else "MLflow is not available"
        except Exception:
            mlflow_status = "unavailable"
            mlflow_message = "MLflow service not initialized"
        
        # Return in format expected by frontend
        return {
            "details": {
                "database": {
                    "status": db_status,
                    "database": db_status,  # Legacy format support
                    "message": db_health.get("message", f"Database is {db_status}"),
                    **db_health
                },
                "kafka": {
                    "status": kafka_status,
                    "kafka": kafka_status,  # Legacy format support
                    "message": f"Kafka is {kafka_status}",
                    "available": kafka_available,
                    "producer_initialized": kafka_service.producer is not None,
                    "library_installed": getattr(kafka_service, "available", False),
                    "active_consumers": len(kafka_service.consumers)
                },
                "rl_environment": {
                    "status": rl_status,
                    "rl_environment": rl_status,  # Legacy format support
                    "message": rl_message
                },
                "mlflow": {
                    "status": mlflow_status,
                    "mlflow": mlflow_status,  # Legacy format support
                    "message": mlflow_message
                }
            },
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
