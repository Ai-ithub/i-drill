from fastapi import APIRouter, HTTPException, Query

from api.models.schemas import (
    DVRProcessRequest,
    DVRProcessResponse,
    DVRStatsResponse,
    DVRAnomalyResponse,
    DVREvaluateRequest,
    DVREvaluateResponse,
)
from services.dvr_service import dvr_service

router = APIRouter(prefix="/dvr", tags=["dvr"])


@router.post("/process", response_model=DVRProcessResponse)
async def process_dvr_record(request: DVRProcessRequest):
    result = dvr_service.process_record(request.record)
    return DVRProcessResponse(**result)


@router.get("/stats", response_model=DVRStatsResponse)
async def get_dvr_stats(limit: int = Query(50, ge=1, le=500)):
    result = dvr_service.get_recent_stats(limit)
    return DVRStatsResponse(**result)


@router.get("/anomalies", response_model=DVRAnomalyResponse)
async def get_dvr_anomaly_snapshot(history_size: int = Query(100, ge=10, le=1000)):
    result = dvr_service.get_anomaly_snapshot(history_size)
    return DVRAnomalyResponse(**result)


@router.post("/evaluate", response_model=DVREvaluateResponse)
async def evaluate_dvr_record(request: DVREvaluateRequest):
    result = dvr_service.evaluate_record_anomaly(request.record, history_size=request.history_size)
    return DVREvaluateResponse(**result)
