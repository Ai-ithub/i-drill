"""
FastAPI Application - Main Entry Point
i-Drill Backend API Server
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime
from uuid import uuid4
import os
from typing import List

try:
    from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

    PROXY_MIDDLEWARE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ProxyHeadersMiddleware = None  # type: ignore
    PROXY_MIDDLEWARE_AVAILABLE = False

# Import database
from database import init_database, check_database_health, db_manager

# Import API routes
from api.routes import (
    health,
    sensor_data,
    predictions,
    maintenance,
    producer,
    config as config_routes,
    auth,
    rl,
    dvr
)

# Import services
from services.data_bridge import DataBridge
from services.auth_service import ensure_default_admin_account, SECRET_KEY

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    RATE_LIMITING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RATE_LIMITING_AVAILABLE = False
    Limiter = None  # type: ignore
    get_remote_address = None  # type: ignore
    RateLimitExceeded = None  # type: ignore
    SlowAPIMiddleware = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development").lower()
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)


def _validate_security_settings(allowed_origins: List[str]) -> None:
    if APP_ENV != "production":
        return

    if SECRET_KEY.startswith("your-secret-key"):
        raise RuntimeError("SECRET_KEY must be configured for production deployments.")

    if set(allowed_origins) == set(DEFAULT_ALLOWED_ORIGINS):
        raise RuntimeError("ALLOWED_ORIGINS must be explicitly set in production.")


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI application
    Handles startup and shutdown procedures
    """
    # ===== STARTUP =====
    logger.info("🚀 Starting i-Drill Backend API Server...")
    
    # Initialize database
    try:
        logger.info("📊 Initializing database connection...")
        db_initialized = init_database(
            database_url=None,  # Uses environment variable or default
            create_tables=True,
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        
        if db_initialized:
            logger.info("✅ Database initialized successfully")
            ensure_default_admin_account()
        else:
            logger.warning("⚠️ Database initialization failed - running in limited mode")
            
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        logger.info("⚠️ Continuing without database (some features will be unavailable)")
    
    # Start Data Bridge (Kafka Consumer → Database → WebSocket)
    try:
        logger.info("🔄 Starting Data Bridge service...")
        data_bridge = DataBridge()
        data_bridge.start()
        app.state.data_bridge = data_bridge
        logger.info("✅ Data Bridge service started")
    except Exception as e:
        logger.error(f"⚠️ Data Bridge startup failed: {e}")
        app.state.data_bridge = None
    
    logger.info("✨ i-Drill Backend API Server is ready!")
    logger.info("📚 API Documentation: http://localhost:8001/docs")
    logger.info("🔍 Health Check: http://localhost:8001/health")
    
    yield  # Server is running
    
    # ===== SHUTDOWN =====
    logger.info("🛑 Shutting down i-Drill Backend API Server...")
    
    # Stop Data Bridge
    if hasattr(app.state, 'data_bridge') and app.state.data_bridge:
        try:
            logger.info("Stopping Data Bridge service...")
            app.state.data_bridge.stop()
            logger.info("✅ Data Bridge service stopped")
        except Exception as e:
            logger.error(f"Error stopping Data Bridge: {e}")
    
    # Close database connections
    try:
        logger.info("Closing database connections...")
        db_manager.close()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    logger.info("👋 Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="i-Drill API",
    description="Intelligent Drilling Rig Automation System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# ==================== Middleware Configuration ====================

# CORS Middleware - Allow frontend to access API
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        ",".join(DEFAULT_ALLOWED_ORIGINS),
    ).split(",")
    if origin.strip()
]

_validate_security_settings(allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Support proxied deployments (when dependency در دسترس باشد)
if PROXY_MIDDLEWARE_AVAILABLE and ProxyHeadersMiddleware is not None:
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Optional HTTPS redirect
if os.getenv("FORCE_HTTPS", "false").lower() == "true":
    app.add_middleware(HTTPSRedirectMiddleware)

# Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

limiter = None
if RATE_LIMITING_AVAILABLE and os.getenv("ENABLE_RATE_LIMIT", "true").lower() != "false":
    default_limit = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
    limiter = Limiter(key_func=get_remote_address, default_limits=[default_limit])  # type: ignore[arg-type]
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)  # type: ignore[arg-type]


# ==================== Request/Response Middleware ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and add processing time"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(
        f"📤 {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# ==================== Rate Limit Handler ====================

if RATE_LIMITING_AVAILABLE:

    @app.exception_handler(RateLimitExceeded)  # type: ignore[misc]
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):  # type: ignore[type-arg]
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "error": "Too Many Requests",
                "message": "Rate limit exceeded. Please try again shortly.",
                "timestamp": datetime.now().isoformat(),
            },
        )


# ==================== Exception Handlers ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    trace_id = str(uuid4())
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={"trace_id": trace_id},
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== API Routes ====================

# Include all API routers
app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["Health"]
)

app.include_router(
    sensor_data.router,
    prefix="/api/v1",
    tags=["Sensor Data"]
)

app.include_router(
    predictions.router,
    prefix="/api/v1",
    tags=["Predictions"]
)

app.include_router(
    maintenance.router,
    prefix="/api/v1",
    tags=["Maintenance"]
)

app.include_router(
    producer.router,
    prefix="/api/v1",
    tags=["Producer"]
)

app.include_router(
    config_routes.router,
    prefix="/api/v1",
    tags=["Configuration"]
)

app.include_router(
    auth.router,
    prefix="/api/v1",
    tags=["Authentication"]
)

app.include_router(
    rl.router,
    prefix="/api/v1",
    tags=["RL"]
)

app.include_router(
    dvr.router,
    prefix="/api/v1",
    tags=["DVR"]
)


# ==================== Root Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "i-Drill API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint"""
    db_health = check_database_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "healthy",
            "database": db_health.get("database", "unknown"),
        },
        "version": "1.0.0"
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
