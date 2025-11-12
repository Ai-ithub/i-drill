"""
Error Response Schemas for API Documentation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ErrorDetail(BaseModel):
    """Error detail information"""
    code: str = Field(..., description="Error code", examples=["VALIDATION_ERROR", "NOT_FOUND"])
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response format"""
    success: bool = Field(False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error information")
    trace_id: str = Field(..., description="Unique trace ID for error tracking")
    timestamp: datetime = Field(..., description="Error timestamp in ISO8601 format")
    path: str = Field(..., description="API path where error occurred")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Request validation failed",
                        "details": {
                            "errors": [
                                {
                                    "field": "body -> depth",
                                    "message": "Input should be greater than or equal to 0",
                                    "type": "greater_than_equal",
                                    "input": -10
                                }
                            ],
                            "count": 1
                        }
                    },
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "path": "/api/v1/sensor-data"
                },
                {
                    "success": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Resource not found with id: rig_001",
                        "details": {
                            "resource": "Rig",
                            "resource_id": "rig_001"
                        }
                    },
                    "trace_id": "550e8400-e29b-41d4-a716-446655440001",
                    "timestamp": "2025-01-15T10:31:00Z",
                    "path": "/api/v1/sensor-data/realtime"
                }
            ]
        }
    }


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    field: str = Field(..., description="Field path where validation failed")
    message: str = Field(..., description="Validation error message")
    type: str = Field(..., description="Validation error type")
    input: Optional[Any] = Field(None, description="Invalid input value")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    error: ErrorDetail = Field(
        ...,
        description="Validation error information",
        examples=[{
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "errors": [
                    {
                        "field": "body -> depth",
                        "message": "Input should be greater than or equal to 0",
                        "type": "greater_than_equal",
                        "input": -10
                    }
                ],
                "count": 1
            }
        }]
    )

