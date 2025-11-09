"""
Predictions API Routes
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.models.schemas import (
    RULPredictionRequest,
    RULPredictionResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    ModelPromotionRequest,
    OperationStatusResponse,
    ModelRegistryEntry,
    ModelRegistryResponse,
    ModelVersionListResponse,
)
from services.prediction_service import PredictionService
from services.data_service import DataService
from services.training_pipeline_service import training_pipeline_service
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])
prediction_service = PredictionService()
data_service = DataService()


@router.post("/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """
    Predict Remaining Useful Life (RUL)
    
    Uses the specified ML model to predict the remaining useful life of drilling equipment.
    """
    try:
        result = prediction_service.predict_rul(
            rig_id=request.rig_id,
            sensor_data=request.sensor_data,
            model_type=request.model_type,
            lookback_window=request.lookback_window
        )
        
        return RULPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error predicting RUL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rul/auto")
async def predict_rul_auto(
    rig_id: str = Query(..., description="Rig ID"),
    lookback_hours: int = Query(24, ge=1, le=168, description="Hours of historical data to use"),
    model_type: str = Query("lstm", pattern="^(lstm|transformer|cnn_lstm)$", description="Model type to use")
):
    """
    Automatically predict RUL using recent data from database
    
    Retrieves recent sensor data from the database and predicts RUL.
    """
    try:
        # Get historical data
        end_time = datetime.now()
        start_time = datetime.now() - timedelta(hours=lookback_hours)
        
        historical_data = data_service.get_historical_data(
            rig_id=rig_id,
            start_time=start_time,
            end_time=end_time,
            limit=200
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data found for rig {rig_id}")
        
        # Predict RUL
        result = prediction_service.predict_rul(
            rig_id=rig_id,
            sensor_data=historical_data,
            model_type=model_type,
            lookback_window=50
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in auto RUL prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anomaly-detection")
async def detect_anomalies(sensor_data: dict):
    """
    Detect anomalies in sensor data
    
    Performs real-time anomaly detection on sensor readings.
    """
    try:
        result = prediction_service.detect_anomalies(sensor_data)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomaly-detection/{rig_id}")
async def get_anomaly_detection_history(
    rig_id: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of history to check")
):
    """
    Get anomaly detection history for a rig
    
    Returns recent anomaly detections for the specified rig.
    """
    try:
        # Get historical data
        end_time = datetime.now()
        start_time = datetime.now() - timedelta(hours=hours)
        
        historical_data = data_service.get_historical_data(
            rig_id=rig_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data found for rig {rig_id}")
        
        # Detect anomalies in all data points
        anomalies = []
        for data_point in historical_data:
            result = prediction_service.detect_anomalies(data_point)
            if result['has_anomaly']:
                anomalies.append({
                    'timestamp': data_point.get('timestamp'),
                    'anomalies': result['anomalies']
                })
        
        return {
            "success": True,
            "rig_id": rig_id,
            "time_range_hours": hours,
            "total_anomalies": len(anomalies),
            "anomalies": anomalies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anomaly history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/train", response_model=TrainingJobResponse)
async def start_training_job(request: TrainingJobRequest):
    result = training_pipeline_service.start_training_job(
        model_name=request.model_name,
        parameters=request.parameters,
        experiment_name=request.experiment_name,
    )

    if not result.get("success"):
        message = result.get("message") or "Training job failed"
        status = 503 if "not configured" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)

    return TrainingJobResponse(**result)


@router.post("/pipeline/promote", response_model=OperationStatusResponse)
async def promote_trained_model(request: ModelPromotionRequest):
    result = training_pipeline_service.promote_model(
        model_name=request.model_name,
        version=request.version,
        stage=request.stage,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message") or "Model promotion failed")

    return OperationStatusResponse(success=True, message="Model promoted successfully")


@router.get("/pipeline/models", response_model=ModelRegistryResponse)
async def list_registered_models():
    models = training_pipeline_service.list_registered_models()
    registry = [ModelRegistryEntry(**model) for model in models]
    return ModelRegistryResponse(success=True, models=registry)


@router.get("/pipeline/models/{model_name}/versions", response_model=ModelVersionListResponse)
async def list_model_versions(model_name: str):
    versions = training_pipeline_service.list_model_versions(model_name)
    return ModelVersionListResponse(success=True, versions=versions)

