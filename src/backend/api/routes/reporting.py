"""
Reporting and Analysis API Routes
For real-time reports, historical analysis, and export capabilities
"""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import StreamingResponse, Response
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from services.reporting_service import reporting_service
from api.dependencies import get_current_user, require_role
from api.models.schemas import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reporting", tags=["Reporting & Analysis"])


# Request Models
class ReportTimeRangeRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    start_time: Optional[str] = Field(None, description="Start time (ISO format)")
    end_time: Optional[str] = Field(None, description="End time (ISO format)")


class ComparisonRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    current_start: str = Field(..., description="Current operation start time (ISO format)")
    current_end: str = Field(..., description="Current operation end time (ISO format)")
    previous_start: Optional[str] = Field(None, description="Previous operation start time (ISO format)")
    previous_end: Optional[str] = Field(None, description="Previous operation end time (ISO format)")


class TrendAnalysisRequest(BaseModel):
    rig_id: str = Field(..., description="Rig identifier")
    start_time: str = Field(..., description="Start time (ISO format)")
    end_time: str = Field(..., description="End time (ISO format)")
    parameter: str = Field("rop", description="Parameter to analyze")
    time_bucket: str = Field("hour", description="Time bucket (hour, day, week)")


@router.post("/performance", response_model=Dict[str, Any])
async def generate_performance_report(
    request: ReportTimeRangeRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate real-time performance report."""
    try:
        start_time = None
        end_time = None
        
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time.replace("Z", "+00:00").split("+")[0])
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.generate_performance_report(
            rig_id=request.rig_id,
            start_time=start_time,
            end_time=end_time
        )
        return result
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/cost", response_model=Dict[str, Any])
async def generate_cost_report(
    request: ReportTimeRangeRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate real-time cost report."""
    try:
        start_time = None
        end_time = None
        
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time.replace("Z", "+00:00").split("+")[0])
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.generate_cost_report(
            rig_id=request.rig_id,
            start_time=start_time,
            end_time=end_time
        )
        return result
    except Exception as e:
        logger.error(f"Error generating cost report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/safety", response_model=Dict[str, Any])
async def generate_safety_report(
    request: ReportTimeRangeRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate real-time safety report."""
    try:
        start_time = None
        end_time = None
        
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time.replace("Z", "+00:00").split("+")[0])
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.generate_safety_report(
            rig_id=request.rig_id,
            start_time=start_time,
            end_time=end_time
        )
        return result
    except Exception as e:
        logger.error(f"Error generating safety report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/compare", response_model=Dict[str, Any])
async def compare_operations(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare current operation with previous operations."""
    try:
        current_start = datetime.fromisoformat(request.current_start.replace("Z", "+00:00").split("+")[0])
        current_end = datetime.fromisoformat(request.current_end.replace("Z", "+00:00").split("+")[0])
        
        previous_start = None
        previous_end = None
        
        if request.previous_start:
            previous_start = datetime.fromisoformat(request.previous_start.replace("Z", "+00:00").split("+")[0])
        if request.previous_end:
            previous_end = datetime.fromisoformat(request.previous_end.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.compare_with_previous_operations(
            rig_id=request.rig_id,
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end
        )
        return result
    except Exception as e:
        logger.error(f"Error comparing operations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/trends", response_model=Dict[str, Any])
async def analyze_trends(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze trends for a parameter over time."""
    try:
        start_time = datetime.fromisoformat(request.start_time.replace("Z", "+00:00").split("+")[0])
        end_time = datetime.fromisoformat(request.end_time.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.analyze_trends(
            rig_id=request.rig_id,
            start_time=start_time,
            end_time=end_time,
            parameter=request.parameter,
            time_bucket=request.time_bucket
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/export/excel")
async def export_report_excel(
    report_type: str = Query(..., description="Report type (performance, cost, safety, comparison, trends)"),
    rig_id: str = Query(..., description="Rig identifier"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    current_user: User = Depends(get_current_user)
):
    """Export report to Excel format."""
    try:
        # Generate report first
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00").split("+")[0])
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00").split("+")[0])
        
        # Generate appropriate report
        if report_type == "performance":
            result = reporting_service.generate_performance_report(rig_id, start_dt, end_dt)
        elif report_type == "cost":
            result = reporting_service.generate_cost_report(rig_id, start_dt, end_dt)
        elif report_type == "safety":
            result = reporting_service.generate_safety_report(rig_id, start_dt, end_dt)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid report type: {report_type}"
            )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to generate report")
            )
        
        # Export to Excel
        excel_data = reporting_service.export_to_excel(
            report_data=result["report"],
            report_type=report_type
        )
        
        filename = f"{report_type}_report_{rig_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return Response(
            content=excel_data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Excel export requires pandas and openpyxl. Please install them."
        )
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/export/pdf")
async def export_report_pdf(
    report_type: str = Query(..., description="Report type (performance, cost, safety, comparison, trends)"),
    rig_id: str = Query(..., description="Rig identifier"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    current_user: User = Depends(get_current_user)
):
    """Export report to PDF format."""
    try:
        # Generate report first
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00").split("+")[0])
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00").split("+")[0])
        
        # Generate appropriate report
        if report_type == "performance":
            result = reporting_service.generate_performance_report(rig_id, start_dt, end_dt)
            title = "Performance Report"
        elif report_type == "cost":
            result = reporting_service.generate_cost_report(rig_id, start_dt, end_dt)
            title = "Cost Report"
        elif report_type == "safety":
            result = reporting_service.generate_safety_report(rig_id, start_dt, end_dt)
            title = "Safety Report"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid report type: {report_type}"
            )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to generate report")
            )
        
        # Export to PDF
        pdf_data = reporting_service.export_to_pdf(
            report_data=result["report"],
            report_type=report_type,
            title=title
        )
        
        filename = f"{report_type}_report_{rig_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF export requires ReportLab. Please install it."
        )
    except Exception as e:
        logger.error(f"Error exporting to PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/export/trends/excel")
async def export_trends_excel(
    rig_id: str = Query(..., description="Rig identifier"),
    start_time: str = Query(..., description="Start time (ISO format)"),
    end_time: str = Query(..., description="End time (ISO format)"),
    parameter: str = Query("rop", description="Parameter to analyze"),
    time_bucket: str = Query("hour", description="Time bucket"),
    current_user: User = Depends(get_current_user)
):
    """Export trend analysis to Excel."""
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00").split("+")[0])
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.analyze_trends(
            rig_id=rig_id,
            start_time=start_dt,
            end_time=end_dt,
            parameter=parameter,
            time_bucket=time_bucket
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to analyze trends")
            )
        
        excel_data = reporting_service.export_to_excel(
            report_data=result["report"],
            report_type="trend_analysis"
        )
        
        filename = f"trends_{parameter}_{rig_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return Response(
            content=excel_data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Excel export requires pandas and openpyxl. Please install them."
        )
    except Exception as e:
        logger.error(f"Error exporting trends to Excel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/export/comparison/excel")
async def export_comparison_excel(
    rig_id: str = Query(..., description="Rig identifier"),
    current_start: str = Query(..., description="Current operation start time (ISO format)"),
    current_end: str = Query(..., description="Current operation end time (ISO format)"),
    previous_start: Optional[str] = Query(None, description="Previous operation start time (ISO format)"),
    previous_end: Optional[str] = Query(None, description="Previous operation end time (ISO format)"),
    current_user: User = Depends(get_current_user)
):
    """Export comparison report to Excel."""
    try:
        current_start_dt = datetime.fromisoformat(current_start.replace("Z", "+00:00").split("+")[0])
        current_end_dt = datetime.fromisoformat(current_end.replace("Z", "+00:00").split("+")[0])
        
        previous_start_dt = None
        previous_end_dt = None
        
        if previous_start:
            previous_start_dt = datetime.fromisoformat(previous_start.replace("Z", "+00:00").split("+")[0])
        if previous_end:
            previous_end_dt = datetime.fromisoformat(previous_end.replace("Z", "+00:00").split("+")[0])
        
        result = reporting_service.compare_with_previous_operations(
            rig_id=rig_id,
            current_start=current_start_dt,
            current_end=current_end_dt,
            previous_start=previous_start_dt,
            previous_end=previous_end_dt
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to compare operations")
            )
        
        excel_data = reporting_service.export_to_excel(
            report_data=result["report"],
            report_type="comparison"
        )
        
        filename = f"comparison_{rig_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return Response(
            content=excel_data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Excel export requires pandas and openpyxl. Please install them."
        )
    except Exception as e:
        logger.error(f"Error exporting comparison to Excel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

