"""
Error Handlers for FastAPI Application
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
from uuid import uuid4
import logging
import traceback
from typing import Any

from api.exceptions import IDrillException

logger = logging.getLogger(__name__)


async def idrill_exception_handler(request: Request, exc: IDrillException) -> JSONResponse:
    """Handle custom i-Drill exceptions"""
    trace_id = str(uuid4())
    
    logger.warning(
        f"i-Drill exception on {request.url.path}: {exc.message}",
        extra={
            "trace_id": trace_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "details": exc.details
        }
    )
    
    response_data = {
        "success": False,
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        },
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }
    
    # Add retry_after header for rate limit errors
    headers = {}
    if exc.error_code == "RATE_LIMIT_EXCEEDED" and "retry_after" in exc.details:
        headers["Retry-After"] = str(exc.details["retry_after"])
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=headers
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors with detailed information"""
    trace_id = str(uuid4())
    
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    logger.warning(
        f"Validation error on {request.url.path}: {errors}",
        extra={"trace_id": trace_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "errors": errors,
                    "count": len(errors)
                }
            },
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions"""
    trace_id = str(uuid4())
    
    # Log full traceback
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={"trace_id": trace_id}
    )
    
    # In production, don't expose internal error details
    import os
    is_production = os.getenv("APP_ENV", "development").lower() == "production"
    
    response_data = {
        "success": False,
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
        },
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }
    
    # Include error details in development
    if not is_production:
        response_data["error"]["details"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc().split("\n")
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )


async def http_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Handle HTTP exceptions"""
    trace_id = str(uuid4())
    
    logger.warning(
        f"HTTP exception on {request.url.path}: {exc.status_code} - {exc.detail}",
        extra={"trace_id": trace_id, "status_code": exc.status_code}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail) if hasattr(exc, 'detail') else "HTTP error occurred"
            },
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

