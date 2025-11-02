"""
FastAPI Backend Application
API endpoints for drilling rig monitoring, data validation, and analytics
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

# Import data quality validation
from src.processing.dvr_controller import process_data, get_validation_report, reset_validator_cache
from src.processing.data_quality_validator import DataQualityReport, ValidationResult

# Import database connections (if they exist)
try:
    from backend.app import test_postgres_connection, test_influx_connection, write_sample_point
except ImportError:
    # Fallback if functions don't exist
    def test_postgres_connection():
        return False
    def test_influx_connection():
        return False
    def write_sample_point():
        return False

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drilling Rig Automation API",
    description="API for real-time drilling rig monitoring, data validation, and analytics",
    version="1.0.0"
)


# Request/Response Models
class SensorData(BaseModel):
    """Model for sensor data validation request"""
    timestamp: str
    rig_id: str
    depth: Optional[float] = None
    wob: Optional[float] = None
    rpm: Optional[float] = None
    torque: Optional[float] = None
    rop: Optional[float] = None
    mud_flow_rate: Optional[float] = None
    mud_pressure: Optional[float] = None
    mud_temperature: Optional[float] = None
    mud_density: Optional[float] = None
    mud_viscosity: Optional[float] = None
    mud_ph: Optional[float] = None
    gamma_ray: Optional[float] = None
    resistivity: Optional[float] = None
    pump_status: Optional[int] = None
    compressor_status: Optional[int] = None
    power_consumption: Optional[float] = None
    vibration_level: Optional[float] = None
    bit_temperature: Optional[float] = None
    motor_temperature: Optional[float] = None
    maintenance_flag: Optional[int] = None
    failure_type: Optional[str] = None


# Health & Status Endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "drilling-rig-api"}


@app.get("/test/postgres")
def test_postgres():
    """Test PostgreSQL connection"""
    try:
        connected = test_postgres_connection()
        return {"postgres_connected": connected, "status": "ok" if connected else "failed"}
    except Exception as e:
        logger.error(f"PostgreSQL test failed: {e}")
        return {"postgres_connected": False, "error": str(e)}


@app.get("/test/influx")
def test_influx():
    """Test InfluxDB connection"""
    try:
        ok = write_sample_point()
        connected = ok and test_influx_connection()
        return {"influx_connected": connected, "status": "ok" if connected else "failed"}
    except Exception as e:
        logger.error(f"InfluxDB test failed: {e}")
        return {"influx_connected": False, "error": str(e)}


# Data Quality Validation Endpoints
@app.post("/api/v1/data-quality/validate")
def validate_sensor_data(data: SensorData):
    """
    Validate sensor data and return detailed quality report
    
    Returns:
        Validation report with quality score, issues, and corrections
    """
    try:
        # Convert Pydantic model to dict
        data_dict = data.model_dump()
        
        # Get validation report
        report = get_validation_report(data_dict)
        
        if report is None:
            raise HTTPException(status_code=500, detail="Failed to generate validation report")
        
        # Format response
        return {
            "timestamp": report.timestamp.isoformat(),
            "rig_id": report.rig_id,
            "overall_score": report.overall_score,
            "is_valid": report.is_valid,
            "missing_fields": report.missing_fields,
            "outlier_fields": report.outlier_fields,
            "issues": [
                {
                    "field_name": r.field_name,
                    "is_valid": r.is_valid,
                    "validation_type": r.validation_type,
                    "message": r.message,
                    "severity": r.severity,
                    "score": r.score if hasattr(r, 'score') else None,
                    "original_value": r.original_value,
                    "corrected_value": r.corrected_value
                }
                for r in report.validation_results
            ],
            "summary": {
                "total_checks": len(report.validation_results),
                "passed_checks": sum(1 for r in report.validation_results if r.is_valid),
                "failed_checks": sum(1 for r in report.validation_results if not r.is_valid),
                "critical_errors": sum(1 for r in report.validation_results if not r.is_valid and r.severity == "error"),
                "warnings": sum(1 for r in report.validation_results if not r.is_valid and r.severity == "warning")
            }
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/api/v1/data-quality/process")
def process_sensor_data(data: SensorData, apply_corrections: bool = True):
    """
    Process sensor data: validate and optionally apply corrections
    
    Args:
        data: Sensor data to process
        apply_corrections: If True, automatically correct fixable issues
    
    Returns:
        Processed data with validation metadata
    """
    try:
        data_dict = data.model_dump()
        
        # Process data
        processed = process_data(data_dict, use_corrections=apply_corrections)
        
        if processed is None:
            raise HTTPException(
                status_code=400,
                detail="Data validation failed - record rejected due to critical errors"
            )
        
        # Remove internal metadata before returning (or keep it if useful)
        validation_metadata = processed.pop("_validation_metadata", {})
        
        return {
            "success": True,
            "data": processed,
            "validation_metadata": validation_metadata,
            "message": "Data processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/data-quality/batch-validate")
def batch_validate_sensor_data(data_list: list[Dict[str, Any]]):
    """
    Validate multiple sensor data records at once
    
    Args:
        data_list: List of sensor data dictionaries
    
    Returns:
        List of validation reports
    """
    try:
        results = []
        for idx, data_dict in enumerate(data_list):
            try:
                report = get_validation_report(data_dict)
                if report:
                    results.append({
                        "index": idx,
                        "rig_id": report.rig_id,
                        "overall_score": report.overall_score,
                        "is_valid": report.is_valid,
                        "missing_fields": report.missing_fields,
                        "outlier_fields": report.outlier_fields,
                        "issue_count": sum(1 for r in report.validation_results if not r.is_valid)
                    })
                else:
                    results.append({
                        "index": idx,
                        "error": "Failed to generate validation report"
                    })
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": str(e)
                })
        
        return {
            "total_records": len(data_list),
            "processed": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")


@app.post("/api/v1/data-quality/reset-cache")
def reset_validation_cache():
    """
    Reset the validator's cache of previous records
    Useful after system restart or when starting fresh
    """
    try:
        reset_validator_cache()
        return {"success": True, "message": "Validation cache reset successfully"}
    except Exception as e:
        logger.error(f"Cache reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache reset failed: {str(e)}")


# Root endpoint
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "name": "Drilling Rig Automation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "data_quality": {
                "validate": "/api/v1/data-quality/validate",
                "process": "/api/v1/data-quality/process",
                "batch_validate": "/api/v1/data-quality/batch-validate",
                "reset_cache": "/api/v1/data-quality/reset-cache"
            }
        }
    }
