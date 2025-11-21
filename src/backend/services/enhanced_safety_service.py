"""
Enhanced Safety Service
Improved real-time safety detection with automatic shutdown, predictive analysis, and enhanced alerts
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import statistics

from services.safety_service import safety_service
from services.drilling_events_service import drilling_events_service
from services.control_service import control_service
from services.scada_connector_service import scada_connector_service
from services.websocket_manager import websocket_manager
from services.data_service import DataService
from api.models.database_models import SafetyEventDB
from database import db_manager

logger = logging.getLogger(__name__)


class EnhancedSafetyService:
    """
    Enhanced Safety Service with real-time monitoring and automatic responses.
    
    Features:
    - Enhanced kick detection with automatic shutdown
    - Predictive stuck pipe detection
    - Emergency stop integration with control system
    - Enhanced formation change detection with alerts
    - Real-time monitoring thread
    """
    
    def __init__(self):
        """Initialize EnhancedSafetyService."""
        self.data_service = DataService()
        self.running = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # Check every second
        
        # Enhanced thresholds
        self.kick_auto_shutdown_enabled = True
        self.kick_auto_shutdown_confidence_threshold = 0.8  # Auto-shutdown if confidence > 80%
        
        # Stuck pipe prediction history
        self.stuck_pipe_prediction_history: Dict[str, deque] = {}  # rig_id -> deque of risk scores
        self.stuck_pipe_prediction_window = 10  # Last 10 predictions
        
        # Emergency stop audit trail
        self.emergency_stop_history: List[Dict[str, Any]] = []
        
        logger.info("Enhanced safety service initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time safety monitoring."""
        if self.running:
            logger.warning("Enhanced safety monitoring is already running")
            return
        
        self.running = True
        import threading
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="EnhancedSafety-Monitoring"
        )
        self.monitoring_thread.start()
        logger.info("Enhanced safety monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time safety monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Enhanced safety monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for real-time safety checks."""
        import time
        
        while self.running:
            try:
                # Get all active rigs (from SCADA connector or data service)
                rigs = self._get_active_rigs()
                
                for rig_id in rigs:
                    try:
                        # Get latest sensor data
                        latest_data = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
                        if not latest_data or len(latest_data) == 0:
                            continue
                        
                        sensor_data = latest_data[0]
                        
                        # 1. Enhanced Kick Detection
                        kick_result = self.detect_kick_enhanced(sensor_data)
                        if kick_result.get("kick_detected") and kick_result.get("auto_shutdown_triggered"):
                            logger.critical(f"🚨 AUTO-SHUTDOWN triggered for rig {rig_id} due to kick detection")
                        
                        # 2. Enhanced Stuck Pipe Detection with Prediction
                        stuck_pipe_result = self.detect_stuck_pipe_enhanced(sensor_data)
                        if stuck_pipe_result.get("stuck_pipe_detected"):
                            logger.warning(f"⚠️ Stuck pipe detected for rig {rig_id}: {stuck_pipe_result.get('risk_level')}")
                        
                        # 3. Enhanced Formation Change Detection
                        formation_result = self.detect_formation_change_enhanced(sensor_data)
                        if formation_result.get("formation_change_detected"):
                            logger.info(f"📊 Formation change detected for rig {rig_id}: {formation_result.get('formation_type')}")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring rig {rig_id}: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(1)
    
    def _get_active_rigs(self) -> List[str]:
        """Get list of active rig IDs."""
        try:
            # Try to get from SCADA connector
            if hasattr(scada_connector_service, 'list_rigs'):
                return scada_connector_service.list_rigs()
            
            # Fallback: get from recent sensor data
            recent_data = self.data_service.get_latest_sensor_data(limit=100)
            rigs = set()
            for record in recent_data:
                rig_id = record.get("rig_id") or record.get("Rig_ID")
                if rig_id:
                    rigs.add(rig_id)
            return list(rigs)
        except Exception as e:
            logger.error(f"Error getting active rigs: {e}")
            return []
    
    def detect_kick_enhanced(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced kick detection with automatic shutdown capability.
        
        Args:
            sensor_data: Current sensor data
        
        Returns:
            Detection result with auto-shutdown status
        """
        rig_id = sensor_data.get("rig_id") or sensor_data.get("Rig_ID", "unknown")
        
        # Use base safety service for detection
        kick_result = safety_service.detect_kick(sensor_data)
        
        if not kick_result.get("kick_detected"):
            return {
                **kick_result,
                "auto_shutdown_triggered": False,
                "auto_shutdown_enabled": self.kick_auto_shutdown_enabled
            }
        
        # Enhanced: Check if auto-shutdown should be triggered
        auto_shutdown_triggered = False
        confidence = kick_result.get("confidence", 0.0)
        severity = kick_result.get("severity", "low")
        
        if (self.kick_auto_shutdown_enabled and 
            confidence >= self.kick_auto_shutdown_confidence_threshold and
            severity in ["critical", "high"]):
            
            # Trigger automatic shutdown
            try:
                shutdown_result = self._trigger_automatic_shutdown(
                    rig_id=rig_id,
                    reason="Kick detected with high confidence",
                    kick_indicators=kick_result.get("indicators", {}),
                    confidence=confidence
                )
                auto_shutdown_triggered = shutdown_result.get("success", False)
                
                if auto_shutdown_triggered:
                    logger.critical(
                        f"🚨 AUTOMATIC SHUTDOWN triggered for rig {rig_id} "
                        f"due to kick detection (confidence: {confidence:.2f})"
                    )
                    
                    # Send critical alert to all users
                    asyncio.create_task(self._send_critical_alert(
                        rig_id=rig_id,
                        alert_type="kick_auto_shutdown",
                        message=f"Automatic shutdown triggered due to kick detection (confidence: {confidence:.2%})",
                        severity="critical",
                        details=kick_result
                    ))
            except Exception as e:
                logger.error(f"Error triggering automatic shutdown: {e}")
        
        return {
            **kick_result,
            "auto_shutdown_triggered": auto_shutdown_triggered,
            "auto_shutdown_enabled": self.kick_auto_shutdown_enabled,
            "auto_shutdown_confidence_threshold": self.kick_auto_shutdown_confidence_threshold
        }
    
    def detect_stuck_pipe_enhanced(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced stuck pipe detection with predictive analysis.
        
        Args:
            sensor_data: Current sensor data
        
        Returns:
            Detection result with prediction probability
        """
        rig_id = sensor_data.get("rig_id") or sensor_data.get("Rig_ID", "unknown")
        
        # Use base safety service for detection
        stuck_pipe_result = safety_service.detect_stuck_pipe(sensor_data)
        
        # Enhanced: Predictive analysis
        risk_score = stuck_pipe_result.get("risk_score", 0.0)
        
        # Update prediction history
        if rig_id not in self.stuck_pipe_prediction_history:
            self.stuck_pipe_prediction_history[rig_id] = deque(maxlen=self.stuck_pipe_prediction_window)
        
        self.stuck_pipe_prediction_history[rig_id].append(risk_score)
        
        # Calculate trend
        history = list(self.stuck_pipe_prediction_history[rig_id])
        if len(history) >= 3:
            # Calculate trend (increasing/decreasing)
            recent_avg = statistics.mean(history[-3:])
            older_avg = statistics.mean(history[:-3]) if len(history) > 3 else recent_avg
            
            trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
            trend_magnitude = abs(recent_avg - older_avg) / max(older_avg, 0.1)
            
            # Predict probability of stuck pipe in next 5 minutes
            if trend == "increasing" and trend_magnitude > 0.2:
                predicted_probability = min(0.95, risk_score + trend_magnitude * 0.3)
            else:
                predicted_probability = risk_score
            
            # Enhanced recommendations based on prediction
            if predicted_probability > 0.7 and not stuck_pipe_result.get("stuck_pipe_detected"):
                # Preemptive recommendations
                enhanced_recommendations = [
                    "⚠️ High risk of stuck pipe predicted",
                    "Reduce WOB immediately as preventive measure",
                    "Increase rotation speed",
                    "Monitor parameters closely",
                    "Consider back-reaming if condition persists"
                ]
                
                stuck_pipe_result["predicted_probability"] = predicted_probability
                stuck_pipe_result["trend"] = trend
                stuck_pipe_result["preemptive_recommendations"] = enhanced_recommendations
                
                # Send warning alert
                asyncio.create_task(self._send_warning_alert(
                    rig_id=rig_id,
                    alert_type="stuck_pipe_prediction",
                    message=f"High probability of stuck pipe predicted ({predicted_probability:.0%})",
                    severity="high",
                    details=stuck_pipe_result
                ))
        else:
            stuck_pipe_result["predicted_probability"] = risk_score
            stuck_pipe_result["trend"] = "insufficient_data"
        
        return stuck_pipe_result
    
    def detect_formation_change_enhanced(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced formation change detection with improved alerts.
        
        Args:
            sensor_data: Current sensor data
        
        Returns:
            Detection result with enhanced recommendations
        """
        # Use base drilling events service
        formation_result = drilling_events_service.detect_formation_change(sensor_data)
        
        if formation_result.get("formation_change_detected"):
            rig_id = sensor_data.get("rig_id") or sensor_data.get("Rig_ID", "unknown")
            formation_type = formation_result.get("formation_type")
            recommended_params = formation_result.get("recommended_parameters", {})
            
            # Enhanced: Send alert with parameter recommendations
            asyncio.create_task(self._send_formation_change_alert(
                rig_id=rig_id,
                formation_type=formation_type,
                depth=formation_result.get("depth"),
                recommended_parameters=recommended_params,
                confidence=formation_result.get("confidence", 0.0)
            ))
        
        return formation_result
    
    def emergency_stop_from_dashboard(
        self,
        rig_id: str,
        reason: str,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute emergency stop from dashboard with full audit trail.
        
        Args:
            rig_id: Rig identifier
            reason: Reason for emergency stop
            user_id: User ID who initiated
            user_name: User name who initiated
            description: Additional description
        
        Returns:
            Emergency stop result with audit information
        """
        try:
            # Execute emergency stop using base service
            stop_result = safety_service.emergency_stop(
                rig_id=rig_id,
                reason=reason,
                description=description,
                user_id=user_id
            )
            
            # Enhanced: Create detailed audit trail
            audit_entry = {
                "rig_id": rig_id,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "description": description,
                "user_id": user_id,
                "user_name": user_name,
                "event_id": stop_result.get("event_id"),
                "actions_taken": stop_result.get("actions_taken", []),
                "source": "dashboard",
                "ip_address": None,  # Could be added from request context
                "user_agent": None   # Could be added from request context
            }
            
            # Store in audit trail
            self.emergency_stop_history.append(audit_entry)
            
            # Keep only last 1000 entries
            if len(self.emergency_stop_history) > 1000:
                self.emergency_stop_history = self.emergency_stop_history[-1000:]
            
            # Also persist to database
            try:
                with db_manager.session_scope() as session:
                    # Update safety event with audit info
                    if stop_result.get("event_id"):
                        event = session.query(SafetyEventDB).filter(
                            SafetyEventDB.id == stop_result["event_id"]
                        ).first()
                        if event:
                            event.metadata = event.metadata or {}
                            event.metadata.update({
                                "audit": audit_entry,
                                "source": "dashboard"
                            })
                            session.commit()
            except Exception as e:
                logger.error(f"Error persisting audit trail: {e}")
            
            logger.critical(
                f"🚨 EMERGENCY STOP from dashboard: rig={rig_id}, "
                f"user={user_name} (ID: {user_id}), reason={reason}"
            )
            
            return {
                **stop_result,
                "audit_entry": audit_entry,
                "source": "dashboard"
            }
        
        except Exception as e:
            logger.error(f"Error executing emergency stop from dashboard: {e}")
            return {
                "success": False,
                "message": f"Failed to execute emergency stop: {str(e)}",
                "event_id": None,
                "timestamp": datetime.now().isoformat(),
                "actions_taken": []
            }
    
    def _trigger_automatic_shutdown(
        self,
        rig_id: str,
        reason: str,
        kick_indicators: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:
        """Trigger automatic shutdown via control system."""
        try:
            actions_taken = []
            
            # 1. Stop RPM
            rpm_result = control_service.apply_parameter_change(
                rig_id=rig_id,
                component="drilling",
                parameter="rpm",
                new_value=0.0,
                metadata={
                    "reason": "automatic_shutdown_kick",
                    "confidence": confidence,
                    "indicators": kick_indicators
                }
            )
            if rpm_result.get("success"):
                actions_taken.append("Stopped RPM to 0")
            
            # 2. Reduce WOB to 0
            wob_result = control_service.apply_parameter_change(
                rig_id=rig_id,
                component="drilling",
                parameter="wob",
                new_value=0.0,
                metadata={
                    "reason": "automatic_shutdown_kick",
                    "confidence": confidence
                }
            )
            if wob_result.get("success"):
                actions_taken.append("Reduced WOB to 0")
            
            # 3. Maintain mud flow (critical for well control)
            # Don't stop mud flow during kick
            
            # 4. Create emergency stop event
            emergency_stop_result = safety_service.emergency_stop(
                rig_id=rig_id,
                reason=f"Automatic shutdown: {reason}",
                description=f"Automatic shutdown triggered due to kick detection (confidence: {confidence:.2%})",
                user_id=None  # System-initiated
            )
            
            return {
                "success": len(actions_taken) >= 2,  # At least RPM and WOB should be stopped
                "actions_taken": actions_taken,
                "emergency_stop_event_id": emergency_stop_result.get("event_id"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error triggering automatic shutdown: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions_taken": [],
                "timestamp": datetime.now().isoformat()
            }
    
    async def _send_critical_alert(
        self,
        rig_id: str,
        alert_type: str,
        message: str,
        severity: str,
        details: Dict[str, Any]
    ) -> None:
        """Send critical alert to all connected users."""
        try:
            alert_message = {
                "message_type": "critical_safety_alert",
                "data": {
                    "alert_type": alert_type,
                    "rig_id": rig_id,
                    "severity": severity,
                    "message": message,
                    "details": details,
                    "timestamp": datetime.now().isoformat(),
                    "requires_acknowledgment": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket_manager.send_to_rig(rig_id, alert_message)
            
            # Also send email/SMS notifications (if configured)
            # This would integrate with email_service
            
        except Exception as e:
            logger.error(f"Error sending critical alert: {e}")
    
    async def _send_warning_alert(
        self,
        rig_id: str,
        alert_type: str,
        message: str,
        severity: str,
        details: Dict[str, Any]
    ) -> None:
        """Send warning alert."""
        try:
            alert_message = {
                "message_type": "safety_warning",
                "data": {
                    "alert_type": alert_type,
                    "rig_id": rig_id,
                    "severity": severity,
                    "message": message,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket_manager.send_to_rig(rig_id, alert_message)
        except Exception as e:
            logger.error(f"Error sending warning alert: {e}")
    
    async def _send_formation_change_alert(
        self,
        rig_id: str,
        formation_type: str,
        depth: Optional[float],
        recommended_parameters: Dict[str, Any],
        confidence: float
    ) -> None:
        """Send formation change alert with recommendations."""
        try:
            alert_message = {
                "message_type": "formation_change_alert",
                "data": {
                    "rig_id": rig_id,
                    "formation_type": formation_type,
                    "depth": depth,
                    "confidence": confidence,
                    "recommended_parameters": recommended_parameters,
                    "message": f"Formation change detected: {formation_type} at depth {depth}m",
                    "timestamp": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket_manager.send_to_rig(rig_id, alert_message)
        except Exception as e:
            logger.error(f"Error sending formation change alert: {e}")
    
    def get_emergency_stop_history(
        self,
        rig_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get emergency stop audit trail."""
        history = self.emergency_stop_history
        
        if rig_id:
            history = [entry for entry in history if entry.get("rig_id") == rig_id]
        
        return history[-limit:] if limit else history
    
    def get_stuck_pipe_prediction(self, rig_id: str) -> Dict[str, Any]:
        """Get stuck pipe prediction statistics for a rig."""
        if rig_id not in self.stuck_pipe_prediction_history:
            return {
                "rig_id": rig_id,
                "prediction_available": False,
                "message": "Insufficient data for prediction"
            }
        
        history = list(self.stuck_pipe_prediction_history[rig_id])
        
        if len(history) < 3:
            return {
                "rig_id": rig_id,
                "prediction_available": False,
                "message": "Insufficient data for prediction",
                "data_points": len(history)
            }
        
        current_risk = history[-1] if history else 0.0
        avg_risk = statistics.mean(history)
        trend = "increasing" if len(history) >= 2 and history[-1] > history[-2] else "decreasing" if len(history) >= 2 and history[-1] < history[-2] else "stable"
        
        # Predict probability
        if trend == "increasing":
            predicted_probability = min(0.95, current_risk + 0.2)
        else:
            predicted_probability = current_risk
        
        return {
            "rig_id": rig_id,
            "prediction_available": True,
            "current_risk_score": current_risk,
            "average_risk_score": avg_risk,
            "trend": trend,
            "predicted_probability": predicted_probability,
            "data_points": len(history),
            "recommendation": "High risk - take preventive action" if predicted_probability > 0.7 else "Monitor closely" if predicted_probability > 0.5 else "Low risk"
        }


# Global instance
enhanced_safety_service = EnhancedSafetyService()

