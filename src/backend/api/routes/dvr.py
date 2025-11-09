from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from api.models.schemas import (
    DVRProcessRequest,
    DVRProcessResponse,
    DVRStatsResponse,
    DVRAnomalyResponse,
    DVREvaluateRequest,
    DVREvaluateResponse,
    DVRHistoryResponse,
    DVRHistoryEntryResponse,
    DVRHistoryUpdateRequest,
    DVRHistoryEntry,
    DVRRecordStatus,
)
from services.dvr_service import dvr_service

router = APIRouter(prefix="/dvr", tags=["dvr"])


@router.post("/process", response_model=DVRProcessResponse)
async def process_dvr_record(request: DVRProcessRequest):
    result = dvr_service.process_record(request.record)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))

    return DVRProcessResponse(
        success=True,
        processed_record=result.get("processed_record"),
        message=result.get("message"),
        history_id=result.get("history_id"),
        anomaly_flag=result.get("anomaly_flag"),
    )


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


@router.get("/history", response_model=DVRHistoryResponse)
async def get_dvr_history(
    limit: int = Query(50, ge=1, le=500),
    rig_id: Optional[str] = Query(None),
    status: Optional[DVRRecordStatus] = Query(None),
):
    history_raw = dvr_service.get_history(limit=limit, rig_id=rig_id, status=status.value if status else None)
    history = [DVRHistoryEntry(**entry) for entry in history_raw]
    return DVRHistoryResponse(success=True, count=len(history), history=history)


@router.patch("/history/{entry_id}", response_model=DVRHistoryEntryResponse)
async def update_dvr_history_entry(entry_id: int, request: DVRHistoryUpdateRequest):
    updates = request.model_dump(exclude_unset=True)
    updated = dvr_service.update_history_entry(entry_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="History entry not found")

    return DVRHistoryEntryResponse(success=True, entry=DVRHistoryEntry(**updated), message="History entry updated")


@router.delete("/history/{entry_id}")
async def delete_dvr_history_entry(entry_id: int):
    deleted = dvr_service.delete_history_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History entry not found")

    return {"success": True, "message": "History entry deleted"}
