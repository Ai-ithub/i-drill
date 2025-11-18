"""
FastAPI Application - Main Entry Point
i-Drill Backend API Server
"""
from fastapi import FastAPI, Request, status, HTTPException
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
    dvr,
    backup,
    control,
    integration,
)

# Import services
from services.data_bridge import DataBridge
from services.auth_service import ensure_default_admin_account

# Import security utilities
from utils.security import get_or_generate_secret_key, validate_cors_origins

# Import error handlers
from api.exceptions import IDrillException
from api.error_handlers import (
    idrill_exception_handler,
    validation_exception_handler,
    general_exception_handler,
    http_exception_handler
)

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
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
)


def _validate_security_settings(allowed_origins: List[str]) -> None:
    """Validate security settings for production environment"""
    if APP_ENV != "production":
        return

    # Validate SECRET_KEY
    # Note: get_or_generate_secret_key() already validates SECRET_KEY and raises
    # RuntimeError if not set or contains insecure patterns. This is a double-check.
    try:
        secret_key = get_or_generate_secret_key()
        
        # Additional validation for production
        if len(secret_key) < 32:
            raise RuntimeError(
                f"SECRET_KEY is too short ({len(secret_key)} chars). "
                "Minimum 32 characters required for production."
            )
    except RuntimeError:
        # Re-raise RuntimeError from get_or_generate_secret_key()
        raise
    except Exception as e:
        raise RuntimeError(f"SECRET_KEY validation failed: {e}")

    # Validate CORS origins
    if set(allowed_origins) == set(DEFAULT_ALLOWED_ORIGINS):
        raise RuntimeError(
            "ALLOWED_ORIGINS must be explicitly set in production. "
            "Cannot use default localhost origins."
        )
    
    # Check for wildcards in production
    for origin in allowed_origins:
        if "*" in origin:
            raise RuntimeError(
                f"CORS origin contains wildcard: {origin}. "
                "Wildcards are not allowed in production."
            )
        if not (origin.startswith("http://") or origin.startswith("https://")):
            raise RuntimeError(
                f"Invalid CORS origin format: {origin}. "
                "Must start with http:// or https://"
            )


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
        # Use use_migrations=True to avoid trying to create tables immediately
        db_initialized = init_database(
            database_url=None,  # Uses environment variable or default
            create_tables=False,  # Don't create tables during startup
            use_migrations=True,  # Use migrations for production
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        
        if db_initialized:
            logger.info("✅ Database initialized successfully")
            try:
                ensure_default_admin_account()
            except Exception as e:
                logger.warning(f"Could not create default admin account: {e}")
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
    
    # Start ML retraining service if enabled
    try:
        from services.ml_retraining_service import ml_retraining_service
        if ml_retraining_service.enabled:
            ml_retraining_service.start()
            app.state.ml_retraining_service = ml_retraining_service
            logger.info("✅ ML retraining service started")
    except Exception as e:
        logger.warning(f"ML retraining service not available: {e}")
    
    # Start backup service if enabled
    try:
        from services.backup_service import backup_service
        if backup_service.enabled:
            backup_service.start()
            app.state.backup_service = backup_service
            logger.info("✅ Backup service started")
    except Exception as e:
        logger.warning(f"Backup service not available: {e}")
    
    logger.info("✨ i-Drill Backend API Server is ready!")
    logger.info("📚 API Documentation: http://localhost:8001/docs")
    logger.info("🔍 Health Check: http://localhost:8001/health")
    logger.info("📊 Metrics: http://localhost:8001/metrics")
    
    yield  # Server is running
    
    # ===== SHUTDOWN =====
    logger.info("🛑 Shutting down i-Drill Backend API Server...")
    
    # Stop ML retraining service
    if hasattr(app.state, 'ml_retraining_service') and app.state.ml_retraining_service:
        try:
            app.state.ml_retraining_service.stop()
            logger.info("✅ ML retraining service stopped")
        except Exception as e:
            logger.error(f"Error stopping ML retraining service: {e}")
    
    # Stop backup service
    if hasattr(app.state, 'backup_service') and app.state.backup_service:
        try:
            app.state.backup_service.stop()
            logger.info("✅ Backup service stopped")
        except Exception as e:
            logger.error(f"Error stopping backup service: {e}")
    
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
    description="""
    ## Intelligent Drilling Rig Automation System API
    
    Comprehensive API for real-time drilling operations monitoring, AI-driven optimization, and predictive maintenance.
    
    ### Features
    
    - 🎛️ **Real-Time Monitoring** - Live sensor data streaming with WebSocket support
    - 🤖 **AI Predictions** - RUL prediction and anomaly detection using ML models
    - 🔧 **Predictive Maintenance** - Maintenance alerts and scheduling
    - 🎮 **Reinforcement Learning** - Automated drilling parameter optimization
    - 📊 **Data Validation & Reconciliation** - DVR system for data quality
    - 🔐 **Authentication** - JWT-based authentication with RBAC
    
    ### Authentication
    
    Most endpoints require authentication. Use `/api/v1/auth/login` to obtain a JWT token.
    
    ### Error Handling
    
    All errors follow a consistent format:
    ```json
    {
        "success": false,
        "error": {
            "code": "ERROR_CODE",
            "message": "Human-readable error message",
            "details": {}
        },
        "trace_id": "uuid",
        "timestamp": "ISO8601",
        "path": "/api/v1/endpoint"
    }
    ```
    
    ### Rate Limiting
    
    API requests are rate-limited. Check response headers for rate limit information.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "i-Drill Support",
        "url": "https://github.com/Ai-ithub/i-drill",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8001",
            "description": "Development server"
        },
        {
            "url": "https://api.i-drill.example.com",
            "description": "Production server"
        }
    ]
)


# ==================== Middleware Configuration ====================

# CORS Middleware - Allow frontend to access API
# Support both ALLOWED_ORIGINS and CORS_ORIGINS for backward compatibility
cors_origins_env = os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins_raw = [
        origin.strip()
        for origin in cors_origins_env.split(",")
        if origin.strip()
    ]
else:
    allowed_origins_raw = list(DEFAULT_ALLOWED_ORIGINS)

# Validate and sanitize CORS origins
allowed_origins = validate_cors_origins(allowed_origins_raw)

# Validate security settings (only in production)
_validate_security_settings(allowed_origins)

# Define allowed methods (restrict for security)
allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
if APP_ENV == "production":
    # In production, be more restrictive
    allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
else:
    # In development, restrict methods for security (even in dev)
    allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

# Define allowed headers (restrict for security)
allowed_headers = [
    "Content-Type",
    "Authorization",
    "Accept",
    "Accept-Language",
    "Content-Language",
    "X-Requested-With",
]
if APP_ENV == "production":
    # In production, use explicit headers only
    pass  # Use the list above
else:
    # In development, restrict headers for security (even in dev)
    # Use explicit headers list instead of "*"
    pass  # Use the list above

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# Support proxied deployments
if PROXY_MIDDLEWARE_AVAILABLE and ProxyHeadersMiddleware is not None:
    # In production, use specific trusted hosts; in development, allow all
    if APP_ENV == "production":
        # In production, get trusted hosts from environment or use specific hosts
        trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",")
        trusted_hosts = [h.strip() for h in trusted_hosts if h.strip()]
        if not trusted_hosts:
            logger.warning("No TRUSTED_HOSTS set in production. Using '*' as fallback.")
            trusted_hosts = "*"
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=trusted_hosts)
    else:
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Optional HTTPS redirect
if os.getenv("FORCE_HTTPS", "false").lower() == "true":
    app.add_middleware(HTTPSRedirectMiddleware)

# Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Import security utilities
from utils.security import get_rate_limit_config, get_security_headers

# Rate Limiting Configuration
rate_limit_config = get_rate_limit_config()

# In production, rate limiting MUST be enabled
if APP_ENV == "production" and not rate_limit_config["enabled"]:
    raise RuntimeError(
        "Rate limiting is mandatory in production. "
        "Set ENABLE_RATE_LIMIT=true in your environment variables."
    )

limiter = None

if RATE_LIMITING_AVAILABLE and rate_limit_config["enabled"]:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.middleware import SlowAPIMiddleware
    
    # Use Redis for rate limiting in production if available
    storage_url = rate_limit_config["storage_url"]
    if storage_url == "memory://" and APP_ENV == "production":
        # Try to use Redis for rate limiting in production
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_password = os.getenv("REDIS_PASSWORD")
        
        if redis_password:
            storage_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
        else:
            storage_url = f"redis://{redis_host}:{redis_port}"
        
        logger.info(f"Using Redis for rate limiting: {redis_host}:{redis_port}")
    elif storage_url == "memory://" and APP_ENV != "production":
        logger.warning("⚠️ Using in-memory rate limiting. For production, use Redis.")
    
    default_limit = rate_limit_config["limits"]["default"]
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[default_limit],
        storage_uri=storage_url
    )  # type: ignore[arg-type]
    app.state.limiter = limiter
    app.state.rate_limit_config = rate_limit_config
    app.add_middleware(SlowAPIMiddleware)  # type: ignore[arg-type]
    logger.info(f"✅ Rate limiting enabled with default limit: {default_limit}")
    logger.info(f"   Auth limit: {rate_limit_config['limits']['auth']}")
    logger.info(f"   Predictions limit: {rate_limit_config['limits']['predictions']}")
    logger.info(f"   Sensor data limit: {rate_limit_config['limits']['sensor_data']}")
elif not RATE_LIMITING_AVAILABLE:
    if APP_ENV == "production":
        raise RuntimeError(
            "Rate limiting library (slowapi) is not installed. "
            "Install it with: pip install slowapi"
        )
    logger.warning("⚠️ Rate limiting library not available. Install slowapi to enable.")
else:
    if APP_ENV == "production":
        raise RuntimeError(
            "Rate limiting is disabled but required in production. "
            "Set ENABLE_RATE_LIMIT=true in your environment variables."
        )
    logger.warning("⚠️ Rate limiting is disabled")


# ==================== Request/Response Middleware ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests, add processing time, and security headers"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Get API URL for CSP (if needed)
    api_url = None
    if APP_ENV == "production":
        # Try to get from environment or construct from request
        api_url = os.getenv("API_URL") or f"{request.url.scheme}://{request.url.netloc}"
    
    # Add comprehensive Security Headers using utility function
    is_production = APP_ENV == "production"
    security_headers = get_security_headers(is_production=is_production, api_url=api_url)
    
    for header_name, header_value in security_headers.items():
        response.headers[header_name] = header_value
    
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

# Register custom exception handlers
app.add_exception_handler(IDrillException, idrill_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


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

app.include_router(
    control.router,
    prefix="/api/v1",
    tags=["Control"]
)

app.include_router(
    backup.router,
    prefix="/api/v1",
    tags=["Backup"]
)

app.include_router(
    integration.router,
    prefix="/api/v1",
    tags=["Integration"]
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


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    from utils.prometheus_metrics import metrics_response
    return metrics_response()


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
