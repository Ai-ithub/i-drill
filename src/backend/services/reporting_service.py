"""
Reporting and Analysis Service
Real-time reports and historical analysis with export capabilities
"""
import logging
import io
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from services.data_service import DataService
from services.performance_metrics_service import performance_metrics_service
from services.realtime_optimization_service import realtime_optimization_service
from services.alert_management_service import alert_management_service
from services.enhanced_safety_service import enhanced_safety_service

logger = logging.getLogger(__name__)

# Try to import required libraries for export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Excel export will be disabled.")

try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF export will be disabled.")


class ReportingService:
    """
    Reporting and Analysis Service.
    
    Features:
    - Real-time Reports (Performance, Cost, Safety)
    - Historical Analysis (Comparison, Trends)
    - Export to Excel/PDF
    """
    
    def __init__(self):
        """Initialize ReportingService."""
        self.data_service = DataService()
        logger.info("Reporting service initialized")
    
    def generate_performance_report(
        self,
        rig_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate real-time performance report.
        
        Args:
            rig_id: Rig identifier
            start_time: Optional start time (default: 24 hours ago)
            end_time: Optional end time (default: now)
        
        Returns:
            Performance report dictionary
        """
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)
            
            # Get current performance metrics
            metrics_result = realtime_optimization_service.calculate_realtime_performance_metrics(rig_id)
            metrics = metrics_result.get("metrics", {}) if metrics_result.get("success") else {}
            
            # Get historical data for analysis
            historical_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            # Calculate statistics
            if historical_data:
                depths = [d.get("depth") or d.get("Depth", 0) for d in historical_data if d.get("depth") or d.get("Depth")]
                rops = [d.get("rop") or d.get("ROP", 0) for d in historical_data if d.get("rop") or d.get("ROP")]
                wob_values = [d.get("wob") or d.get("WOB", 0) for d in historical_data if d.get("wob") or d.get("WOB")]
                rpm_values = [d.get("rpm") or d.get("RPM", 0) for d in historical_data if d.get("rpm") or d.get("RPM")]
                
                stats = {
                    "total_records": len(historical_data),
                    "depth_range": {
                        "min": min(depths) if depths else 0,
                        "max": max(depths) if depths else 0,
                        "current": depths[-1] if depths else 0
                    },
                    "rop_stats": {
                        "min": min(rops) if rops else 0,
                        "max": max(rops) if rops else 0,
                        "average": sum(rops) / len(rops) if rops else 0,
                        "current": rops[-1] if rops else 0
                    },
                    "wob_stats": {
                        "min": min(wob_values) if wob_values else 0,
                        "max": max(wob_values) if wob_values else 0,
                        "average": sum(wob_values) / len(wob_values) if wob_values else 0,
                        "current": wob_values[-1] if wob_values else 0
                    },
                    "rpm_stats": {
                        "min": min(rpm_values) if rpm_values else 0,
                        "max": max(rpm_values) if rpm_values else 0,
                        "average": sum(rpm_values) / len(rpm_values) if rpm_values else 0,
                        "current": rpm_values[-1] if rpm_values else 0
                    }
                }
            else:
                stats = {}
            
            # Get performance metrics
            rop_efficiency = metrics.get("rop_efficiency", {}).get("current", 0)
            energy_efficiency = metrics.get("energy_efficiency", {}).get("current", 0)
            dei = metrics.get("drilling_efficiency_index", {}).get("current", 0)
            
            report = {
                "report_type": "performance",
                "rig_id": rig_id,
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                },
                "current_metrics": {
                    "rop_efficiency": rop_efficiency,
                    "energy_efficiency": energy_efficiency,
                    "drilling_efficiency_index": dei
                },
                "statistics": stats,
                "summary": {
                    "status": "operational" if rop_efficiency > 70 else "needs_attention",
                    "key_insights": [
                        f"ROP Efficiency: {rop_efficiency:.1f}%",
                        f"Energy Efficiency: {energy_efficiency:.2f} m/kWh",
                        f"Drilling Efficiency Index: {dei:.1f}"
                    ]
                }
            }
            
            return {
                "success": True,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                "success": False,
                "message": str(e),
                "report": None
            }
    
    def generate_cost_report(
        self,
        rig_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate real-time cost report.
        
        Args:
            rig_id: Rig identifier
            start_time: Optional start time (default: 30 days ago)
            end_time: Optional end time (default: now)
        
        Returns:
            Cost report dictionary
        """
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=30)
            
            # Get cost data
            cost_result = realtime_optimization_service.calculate_realtime_cost(rig_id)
            cost_data = cost_result.get("cost", {}) if cost_result.get("success") else {}
            
            # Get historical data for breakdown
            historical_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            # Calculate cost breakdown
            if historical_data:
                # Time-based costs
                total_minutes = len(historical_data)
                time_cost = total_minutes * 10.0  # $10 per minute
                
                # Energy costs
                total_energy = sum([
                    (d.get("power_consumption", 0) or d.get("Power_Consumption", 0)) / 60.0
                    for d in historical_data
                ])
                energy_cost = total_energy * 0.15  # $0.15 per kWh
                
                # Depth drilled
                if historical_data:
                    initial_depth = historical_data[0].get("depth") or historical_data[0].get("Depth", 0)
                    current_depth = historical_data[-1].get("depth") or historical_data[-1].get("Depth", 0)
                    depth_drilled = current_depth - initial_depth
                else:
                    depth_drilled = 0
                
                cost_breakdown = {
                    "time_cost": time_cost,
                    "energy_cost": energy_cost,
                    "total_cost": time_cost + energy_cost,
                    "depth_drilled": depth_drilled,
                    "cost_per_meter": (time_cost + energy_cost) / max(depth_drilled, 0.1),
                    "total_minutes": total_minutes,
                    "total_energy_kwh": total_energy
                }
            else:
                cost_breakdown = {}
            
            report = {
                "report_type": "cost",
                "rig_id": rig_id,
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_days": (end_time - start_time).total_seconds() / 86400
                },
                "current_cost": cost_data,
                "cost_breakdown": cost_breakdown,
                "budget_status": {
                    "budget": cost_data.get("budget", 1000000.0),
                    "total_cost": cost_data.get("total_cost", 0),
                    "remaining_budget": cost_data.get("remaining_budget", 0),
                    "utilization_percent": cost_data.get("budget_utilization", 0),
                    "status": "within_budget" if cost_data.get("budget_utilization", 0) < 80 else "approaching_budget" if cost_data.get("budget_utilization", 0) < 95 else "over_budget"
                },
                "summary": {
                    "key_insights": [
                        f"Total Cost: ${cost_data.get('total_cost', 0):,.2f}",
                        f"Cost per Meter: ${cost_data.get('cost_per_meter', 0):,.2f}",
                        f"Budget Utilization: {cost_data.get('budget_utilization', 0):.1f}%"
                    ]
                }
            }
            
            return {
                "success": True,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return {
                "success": False,
                "message": str(e),
                "report": None
            }
    
    def generate_safety_report(
        self,
        rig_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate real-time safety report.
        
        Args:
            rig_id: Rig identifier
            start_time: Optional start time (default: 7 days ago)
            end_time: Optional end time (default: now)
        
        Returns:
            Safety report dictionary
        """
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=7)
            
            # Get active alerts
            active_alerts = alert_management_service.get_active_alerts(rig_id=rig_id)
            
            # Get alert history
            alert_history = alert_management_service.get_alert_history(
                rig_id=rig_id,
                limit=1000
            )
            
            # Filter by time range
            filtered_history = [
                alert for alert in alert_history
                if start_time <= datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00").split("+")[0]) <= end_time
            ]
            
            # Categorize alerts
            alerts_by_severity = defaultdict(int)
            alerts_by_type = defaultdict(int)
            
            for alert in filtered_history:
                severity = alert.get("severity", "unknown")
                alert_type = alert.get("alert_type", "unknown")
                alerts_by_severity[severity] += 1
                alerts_by_type[alert_type] += 1
            
            # Get safety incidents (kicks, stuck pipe, etc.)
            safety_incidents = [
                alert for alert in filtered_history
                if alert.get("alert_type") in ["safety", "kick", "stuck_pipe", "emergency_stop"]
            ]
            
            # Get emergency stops
            emergency_stops = [
                alert for alert in filtered_history
                if alert.get("alert_type") == "emergency_stop"
            ]
            
            report = {
                "report_type": "safety",
                "rig_id": rig_id,
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_days": (end_time - start_time).total_seconds() / 86400
                },
                "current_status": {
                    "active_alerts_count": len(active_alerts),
                    "critical_alerts": len([a for a in active_alerts if a.get("severity") == "critical"]),
                    "high_alerts": len([a for a in active_alerts if a.get("severity") == "high"]),
                    "status": "safe" if len(active_alerts) == 0 else "attention_required" if len([a for a in active_alerts if a.get("severity") == "critical"]) == 0 else "critical"
                },
                "incidents_summary": {
                    "total_incidents": len(safety_incidents),
                    "emergency_stops": len(emergency_stops),
                    "alerts_by_severity": dict(alerts_by_severity),
                    "alerts_by_type": dict(alerts_by_type)
                },
                "recent_incidents": safety_incidents[:10],  # Last 10 incidents
                "summary": {
                    "key_insights": [
                        f"Active Alerts: {len(active_alerts)}",
                        f"Critical Alerts: {len([a for a in active_alerts if a.get('severity') == 'critical'])}",
                        f"Emergency Stops: {len(emergency_stops)}",
                        f"Total Incidents: {len(safety_incidents)}"
                    ],
                    "recommendations": [
                        "Review critical alerts immediately" if len([a for a in active_alerts if a.get("severity") == "critical"]) > 0 else "No critical alerts",
                        "Conduct safety review" if len(safety_incidents) > 5 else "Safety status normal"
                    ]
                }
            }
            
            return {
                "success": True,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error generating safety report: {e}")
            return {
                "success": False,
                "message": str(e),
                "report": None
            }
    
    def compare_with_previous_operations(
        self,
        rig_id: str,
        current_start: datetime,
        current_end: datetime,
        previous_start: Optional[datetime] = None,
        previous_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compare current operation with previous operations.
        
        Args:
            rig_id: Rig identifier
            current_start: Current operation start time
            current_end: Current operation end time
            previous_start: Previous operation start time (default: same duration before current_start)
            previous_end: Previous operation end time (default: current_start)
        
        Returns:
            Comparison report dictionary
        """
        try:
            if previous_end is None:
                previous_end = current_start
            if previous_start is None:
                duration = (current_end - current_start).total_seconds()
                previous_start = previous_end - timedelta(seconds=duration)
            
            # Get current operation data
            current_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=current_start,
                end_time=current_end,
                limit=10000
            )
            
            # Get previous operation data
            previous_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=previous_start,
                end_time=previous_end,
                limit=10000
            )
            
            # Calculate metrics for current
            current_metrics = self._calculate_operation_metrics(current_data)
            
            # Calculate metrics for previous
            previous_metrics = self._calculate_operation_metrics(previous_data)
            
            # Calculate differences
            differences = {}
            for key in current_metrics:
                if key in previous_metrics and previous_metrics[key] != 0:
                    diff_percent = ((current_metrics[key] - previous_metrics[key]) / previous_metrics[key]) * 100
                    differences[key] = {
                        "current": current_metrics[key],
                        "previous": previous_metrics[key],
                        "difference": current_metrics[key] - previous_metrics[key],
                        "difference_percent": diff_percent,
                        "trend": "improved" if diff_percent > 0 else "degraded" if diff_percent < 0 else "stable"
                    }
            
            report = {
                "report_type": "comparison",
                "rig_id": rig_id,
                "generated_at": datetime.now().isoformat(),
                "current_operation": {
                    "start": current_start.isoformat(),
                    "end": current_end.isoformat(),
                    "metrics": current_metrics
                },
                "previous_operation": {
                    "start": previous_start.isoformat(),
                    "end": previous_end.isoformat(),
                    "metrics": previous_metrics
                },
                "comparison": differences,
                "summary": {
                    "overall_trend": "improved" if sum([d.get("difference_percent", 0) for d in differences.values()]) > 0 else "degraded",
                    "key_improvements": [
                        k for k, v in differences.items()
                        if v.get("difference_percent", 0) > 5
                    ],
                    "key_degradations": [
                        k for k, v in differences.items()
                        if v.get("difference_percent", 0) < -5
                    ]
                }
            }
            
            return {
                "success": True,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error comparing operations: {e}")
            return {
                "success": False,
                "message": str(e),
                "report": None
            }
    
    def _calculate_operation_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for an operation."""
        if not data:
            return {}
        
        depths = [d.get("depth") or d.get("Depth", 0) for d in data if d.get("depth") or d.get("Depth")]
        rops = [d.get("rop") or d.get("ROP", 0) for d in data if d.get("rop") or d.get("ROP")]
        wob_values = [d.get("wob") or d.get("WOB", 0) for d in data if d.get("wob") or d.get("WOB")]
        rpm_values = [d.get("rpm") or d.get("RPM", 0) for d in data if d.get("rpm") or d.get("RPM")]
        
        if depths:
            depth_drilled = max(depths) - min(depths)
        else:
            depth_drilled = 0
        
        return {
            "total_records": len(data),
            "depth_drilled": depth_drilled,
            "average_rop": sum(rops) / len(rops) if rops else 0,
            "max_rop": max(rops) if rops else 0,
            "average_wob": sum(wob_values) / len(wob_values) if wob_values else 0,
            "average_rpm": sum(rpm_values) / len(rpm_values) if rpm_values else 0,
            "duration_hours": len(data) / 60.0  # Assuming 1 record per minute
        }
    
    def analyze_trends(
        self,
        rig_id: str,
        start_time: datetime,
        end_time: datetime,
        parameter: str = "rop",
        time_bucket: str = "hour"  # hour, day, week
    ) -> Dict[str, Any]:
        """
        Analyze trends for a parameter over time.
        
        Args:
            rig_id: Rig identifier
            start_time: Start time
            end_time: End time
            parameter: Parameter to analyze (rop, wob, rpm, etc.)
            time_bucket: Time bucket for aggregation (hour, day, week)
        
        Returns:
            Trend analysis dictionary
        """
        try:
            # Get historical data
            historical_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=50000
            )
            
            if not historical_data:
                return {
                    "success": False,
                    "message": "No data available for trend analysis",
                    "trends": None
                }
            
            # Aggregate by time bucket
            if time_bucket == "hour":
                bucket_seconds = 3600
            elif time_bucket == "day":
                bucket_seconds = 86400
            elif time_bucket == "week":
                bucket_seconds = 604800
            else:
                bucket_seconds = 3600
            
            # Group data by time bucket
            buckets = defaultdict(list)
            for record in historical_data:
                timestamp = datetime.fromisoformat(
                    record.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00").split("+")[0]
                )
                bucket_key = int(timestamp.timestamp() / bucket_seconds) * bucket_seconds
                buckets[bucket_key].append(record)
            
            # Calculate aggregated values
            trend_data = []
            for bucket_time in sorted(buckets.keys()):
                bucket_records = buckets[bucket_time]
                values = [
                    r.get(parameter) or r.get(parameter.upper()) or 0
                    for r in bucket_records
                    if r.get(parameter) or r.get(parameter.upper())
                ]
                
                if values:
                    trend_data.append({
                        "timestamp": datetime.fromtimestamp(bucket_time).isoformat(),
                        "bucket": time_bucket,
                        "count": len(bucket_records),
                        "min": min(values),
                        "max": max(values),
                        "average": sum(values) / len(values),
                        "median": sorted(values)[len(values) // 2] if values else 0
                    })
            
            # Calculate trend direction
            if len(trend_data) >= 2:
                first_avg = trend_data[0]["average"]
                last_avg = trend_data[-1]["average"]
                trend_direction = "increasing" if last_avg > first_avg else "decreasing" if last_avg < first_avg else "stable"
                trend_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
            else:
                trend_direction = "insufficient_data"
                trend_percent = 0
            
            report = {
                "report_type": "trend_analysis",
                "rig_id": rig_id,
                "parameter": parameter,
                "time_bucket": time_bucket,
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "trend_data": trend_data,
                "trend_summary": {
                    "direction": trend_direction,
                    "change_percent": trend_percent,
                    "data_points": len(trend_data),
                    "overall_average": sum([d["average"] for d in trend_data]) / len(trend_data) if trend_data else 0
                }
            }
            
            return {
                "success": True,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {
                "success": False,
                "message": str(e),
                "trends": None
            }
    
    def export_to_excel(
        self,
        report_data: Dict[str, Any],
        report_type: str
    ) -> bytes:
        """
        Export report to Excel format.
        
        Args:
            report_data: Report data dictionary
            report_type: Type of report (performance, cost, safety, comparison, trends)
        
        Returns:
            Excel file content as bytes
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel export")
        
        try:
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Main summary sheet
                summary_data = []
                if "summary" in report_data:
                    for key, value in report_data["summary"].items():
                        if isinstance(value, list):
                            summary_data.append({"Metric": key, "Value": ", ".join(str(v) for v in value)})
                        else:
                            summary_data.append({"Metric": key, "Value": str(value)})
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name="Summary", index=False)
                
                # Time range sheet
                if "time_range" in report_data:
                    time_data = [{"Field": k, "Value": str(v)} for k, v in report_data["time_range"].items()]
                    df_time = pd.DataFrame(time_data)
                    df_time.to_excel(writer, sheet_name="Time Range", index=False)
                
                # Report-specific sheets
                if report_type == "trend_analysis" and "trend_data" in report_data:
                    df_trends = pd.DataFrame(report_data["trend_data"])
                    df_trends.to_excel(writer, sheet_name="Trends", index=False)
                
                elif report_type == "comparison" and "comparison" in report_data:
                    comparison_data = []
                    for key, value in report_data["comparison"].items():
                        comparison_data.append({
                            "Metric": key,
                            "Current": value.get("current", 0),
                            "Previous": value.get("previous", 0),
                            "Difference": value.get("difference", 0),
                            "Difference %": value.get("difference_percent", 0),
                            "Trend": value.get("trend", "stable")
                        })
                    df_comparison = pd.DataFrame(comparison_data)
                    df_comparison.to_excel(writer, sheet_name="Comparison", index=False)
                
                elif report_type in ["performance", "cost", "safety"]:
                    # Add statistics or breakdown sheets
                    if "statistics" in report_data:
                        stats_data = []
                        for key, value in report_data["statistics"].items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    stats_data.append({"Category": key, "Metric": sub_key, "Value": sub_value})
                            else:
                                stats_data.append({"Category": "General", "Metric": key, "Value": value})
                        if stats_data:
                            df_stats = pd.DataFrame(stats_data)
                            df_stats.to_excel(writer, sheet_name="Statistics", index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def export_to_pdf(
        self,
        report_data: Dict[str, Any],
        report_type: str,
        title: str = "Drilling Operations Report"
    ) -> bytes:
        """
        Export report to PDF format.
        
        Args:
            report_data: Report data dictionary
            report_type: Type of report
            title: Report title
        
        Returns:
            PDF file content as bytes
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF export")
        
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), title=title)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph(title, styles["Title"]))
            story.append(Spacer(1, 12))
            
            # Report metadata
            metadata = [
                ["Report Type", report_type],
                ["Generated At", report_data.get("generated_at", datetime.now().isoformat())],
                ["Rig ID", report_data.get("rig_id", "N/A")]
            ]
            
            if "time_range" in report_data:
                metadata.append(["Time Range", f"{report_data['time_range'].get('start', 'N/A')} to {report_data['time_range'].get('end', 'N/A')}"])
            
            table = Table(metadata, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            
            # Summary section
            if "summary" in report_data:
                story.append(Paragraph("Summary", styles["Heading2"]))
                summary = report_data["summary"]
                if "key_insights" in summary:
                    for insight in summary["key_insights"]:
                        story.append(Paragraph(f"• {insight}", styles["Normal"]))
                story.append(Spacer(1, 12))
            
            # Report-specific content
            if report_type == "trend_analysis" and "trend_data" in report_data:
                story.append(Paragraph("Trend Data", styles["Heading2"]))
                trend_data = report_data["trend_data"]
                if trend_data:
                    headers = ["Timestamp", "Average", "Min", "Max", "Count"]
                    data = [headers]
                    for item in trend_data[:50]:  # Limit to 50 rows
                        data.append([
                            item.get("timestamp", "N/A")[:19],
                            f"{item.get('average', 0):.2f}",
                            f"{item.get('min', 0):.2f}",
                            f"{item.get('max', 0):.2f}",
                            str(item.get("count", 0))
                        ])
                    table = Table(data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
            
            elif report_type == "comparison" and "comparison" in report_data:
                story.append(Paragraph("Comparison", styles["Heading2"]))
                comparison = report_data["comparison"]
                headers = ["Metric", "Current", "Previous", "Difference", "Difference %", "Trend"]
                data = [headers]
                for key, value in list(comparison.items())[:20]:  # Limit to 20 rows
                    data.append([
                        key,
                        f"{value.get('current', 0):.2f}",
                        f"{value.get('previous', 0):.2f}",
                        f"{value.get('difference', 0):.2f}",
                        f"{value.get('difference_percent', 0):.2f}%",
                        value.get("trend", "stable")
                    ])
                table = Table(data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            raise


# Global instance
reporting_service = ReportingService()

