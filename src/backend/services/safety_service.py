"""
Safety Service
Handles safety-related operations including emergency stop, kick detection, and stuck pipe detection
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from enum import Enum

from api.models.database_models import SafetyEventDB, SensorData
from database import db_manager
from services.control_service import control_service
try:
    from services.websocket_manager import websocket_manager
except ImportError:
    # Fallback if websocket_manager not available
    websocket_manager = None

logger = logging.getLogger(__name__)


class SafetyEventType(str, Enum):
    """Safety event types"""
    EMERGENCY_STOP = "emergency_stop"
    KICK = "kick"
    STUCK_PIPE = "stuck_pipe"
    BLOWOUT = "blowout"
    EQUIPMENT_FAILURE = "equipment_failure"


class EventSeverity(str, Enum):
    """Event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SafetyService:
    """Service for safety operations"""
    
    def __init__(self):
        """Initialize SafetyService"""
        # Detection thresholds
        self.KICK_FLOW_DIFFERENTIAL_THRESHOLD = 50.0  # gpm
        self.KICK_PIT_VOLUME_CHANGE_THRESHOLD = 10.0  # bbl
        self.STUCK_PIPE_ROP_DECREASE_THRESHOLD = 0.3  # 30% decrease
        self.STUCK_PIPE_TORQUE_INCREASE_THRESHOLD = 1.5  # 50% increase
        self.STUCK_PIPE_HOOK_LOAD_DECREASE_THRESHOLD = 0.2  # 20% decrease
        
    def emergency_stop(
        self,
        rig_id: str,
        reason: str,
        description: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute emergency stop for drilling operations
        
        Args:
            rig_id: Rig identifier
            reason: Reason for emergency stop
            description: Additional description
            user_id: User ID who initiated the stop
            
        Returns:
            Dictionary with stop result and event ID
        """
        try:
            logger.critical(f"🚨 EMERGENCY STOP initiated for rig {rig_id} by user {user_id}: {reason}")
            
            # Get current sensor data snapshot
            sensor_snapshot = self._get_current_sensor_data(rig_id)
            
            # Actions to take
            actions_taken = []
            
            # 1. Stop all drilling parameters via control system
            try:
                if control_service.is_available():
                    # Stop RPM
                    control_service.apply_parameter_change(
                        rig_id=rig_id,
                        component="drilling",
                        parameter="rpm",
                        new_value=0.0,
                        metadata={"reason": "emergency_stop", "user_id": user_id}
                    )
                    actions_taken.append("Stopped RPM to 0")
                    
                    # Reduce WOB
                    control_service.apply_parameter_change(
                        rig_id=rig_id,
                        component="drilling",
                        parameter="wob",
                        new_value=0.0,
                        metadata={"reason": "emergency_stop", "user_id": user_id}
                    )
                    actions_taken.append("Reduced WOB to 0")
                    
                    # Maintain mud flow for safety
                    actions_taken.append("Maintained mud flow for well control")
            except Exception as e:
                logger.error(f"Error applying emergency stop to control system: {e}")
                actions_taken.append(f"Control system action failed: {str(e)}")
            
            # 2. Broadcast emergency stop to all WebSocket clients
            try:
                if websocket_manager:
                    emergency_message = {
                        "message_type": "emergency_stop",
                        "data": {
                            "rig_id": rig_id,
                            "reason": reason,
                            "description": description,
                            "timestamp": datetime.now().isoformat(),
                            "severity": "critical"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    websocket_manager.send_to_rig(rig_id, emergency_message)
                    actions_taken.append("Broadcasted emergency stop to all connected clients")
            except Exception as e:
                logger.error(f"Error broadcasting emergency stop: {e}")
            
            # 3. Create safety event record
            with db_manager.session_scope() as session:
                event = SafetyEventDB(
                    rig_id=rig_id,
                    event_type=SafetyEventType.EMERGENCY_STOP.value,
                    severity=EventSeverity.CRITICAL.value,
                    status="active",
                    reason=reason,
                    description=description,
                    sensor_data_snapshot=sensor_snapshot,
                    actions_taken=actions_taken,
                    recommendations=[
                        "Verify all systems are stopped",
                        "Check well control status",
                        "Notify supervisor immediately",
                        "Do not resume until safety clearance"
                    ],
                    created_by=user_id
                )
                session.add(event)
                session.commit()
                session.refresh(event)
                
                logger.critical(f"Emergency stop event recorded: ID={event.id}")
                
                return {
                    "success": True,
                    "message": "Emergency stop executed successfully",
                    "event_id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "actions_taken": actions_taken
                }
                
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
            return {
                "success": False,
                "message": f"Failed to execute emergency stop: {str(e)}",
                "event_id": None,
                "timestamp": datetime.now().isoformat(),
                "actions_taken": []
            }
    
    def detect_kick(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect kick (gas influx) based on sensor data
        
        Args:
            sensor_data: Current sensor data dictionary
            
        Returns:
            Dictionary with detection result
        """
        try:
            rig_id = sensor_data.get('rig_id')
            if not rig_id:
                return {"kick_detected": False, "error": "rig_id missing"}
            
            # Get historical data for comparison
            history = self._get_recent_sensor_data(rig_id, minutes=5)
            if not history:
                return {"kick_detected": False, "error": "insufficient history"}
            
            # Calculate indicators
            flow_in = sensor_data.get('mud_flow', 0) or sensor_data.get('flow_in', 0)
            flow_out = sensor_data.get('flow_out', 0)
            standpipe_pressure = sensor_data.get('mud_pressure', 0) or sensor_data.get('standpipe_pressure', 0)
            mud_weight = sensor_data.get('mud_density', 0)
            
            # Calculate flow differential
            flow_differential = flow_out - flow_in if flow_out and flow_in else 0
            
            # Calculate pit volume change (if available)
            pit_volume_change = self._calculate_pit_volume_change(sensor_data, history)
            
            # Calculate pressure change
            avg_pressure = sum([h.get('mud_pressure', 0) or h.get('standpipe_pressure', 0) for h in history]) / len(history) if history else 0
            pressure_change = standpipe_pressure - avg_pressure if standpipe_pressure and avg_pressure else 0
            
            # Detection logic
            indicators = {
                "flow_differential": flow_differential,
                "flow_differential_threshold": self.KICK_FLOW_DIFFERENTIAL_THRESHOLD,
                "pit_volume_change": pit_volume_change,
                "pit_volume_threshold": self.KICK_PIT_VOLUME_CHANGE_THRESHOLD,
                "pressure_change": pressure_change,
                "standpipe_pressure": standpipe_pressure
            }
            
            kick_detected = False
            severity = "low"
            confidence = 0.0
            immediate_actions = []
            
            # High confidence: Flow differential exceeds threshold
            if flow_differential > self.KICK_FLOW_DIFFERENTIAL_THRESHOLD:
                kick_detected = True
                severity = "critical"
                confidence = min(0.9, 0.5 + (flow_differential / self.KICK_FLOW_DIFFERENTIAL_THRESHOLD) * 0.4)
                immediate_actions = [
                    "STOP DRILLING IMMEDIATELY",
                    "Close BOP (Blowout Preventer)",
                    "Increase mud weight",
                    "Notify supervisor and well control team",
                    "Monitor pit volume continuously"
                ]
            # Medium confidence: Pit volume increase
            elif pit_volume_change > self.KICK_PIT_VOLUME_CHANGE_THRESHOLD:
                kick_detected = True
                severity = "high"
                confidence = 0.6
                immediate_actions = [
                    "Stop drilling",
                    "Monitor flow rates closely",
                    "Check BOP status",
                    "Prepare to increase mud weight",
                    "Notify supervisor"
                ]
            # Low confidence: Pressure increase
            elif pressure_change > 200:  # 200 psi increase
                kick_detected = True
                severity = "medium"
                confidence = 0.4
                immediate_actions = [
                    "Monitor pressure closely",
                    "Check flow rates",
                    "Notify supervisor",
                    "Prepare for well control procedures"
                ]
            
            if kick_detected:
                # Create safety event
                event_id = self._create_safety_event(
                    rig_id=rig_id,
                    event_type=SafetyEventType.KICK.value,
                    severity=severity,
                    sensor_data_snapshot=sensor_data,
                    indicators=indicators,
                    recommendations=immediate_actions
                )
                
                # Broadcast alert
                try:
                    if websocket_manager:
                        alert_message = {
                            "message_type": "safety_alert",
                            "data": {
                                "event_type": "kick",
                                "rig_id": rig_id,
                                "severity": severity,
                                "confidence": confidence,
                                "indicators": indicators,
                                "immediate_actions": immediate_actions,
                                "timestamp": datetime.now().isoformat()
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        websocket_manager.send_to_rig(rig_id, alert_message)
                except Exception as e:
                    logger.error(f"Error broadcasting kick alert: {e}")
                
                logger.warning(f"⚠️ KICK DETECTED for rig {rig_id}: severity={severity}, confidence={confidence:.2f}")
                
                return {
                    "kick_detected": True,
                    "severity": severity,
                    "confidence": confidence,
                    "indicators": indicators,
                    "immediate_actions": immediate_actions,
                    "event_id": event_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "kick_detected": False,
                    "severity": "low",
                    "confidence": 0.0,
                    "indicators": indicators,
                    "immediate_actions": [],
                    "event_id": None,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error detecting kick: {e}")
            return {
                "kick_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def detect_stuck_pipe(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect stuck pipe condition
        
        Args:
            sensor_data: Current sensor data dictionary
            
        Returns:
            Dictionary with detection result
        """
        try:
            rig_id = sensor_data.get('rig_id')
            if not rig_id:
                return {"stuck_pipe_detected": False, "error": "rig_id missing"}
            
            # Get historical data for comparison
            history = self._get_recent_sensor_data(rig_id, minutes=10)
            if not history or len(history) < 5:
                return {"stuck_pipe_detected": False, "error": "insufficient history"}
            
            # Current values
            current_rop = sensor_data.get('rop', 0)
            current_torque = sensor_data.get('torque', 0)
            current_hook_load = sensor_data.get('hook_load', 0)
            current_wob = sensor_data.get('wob', 0)
            current_vibration = sensor_data.get('vibration', 0) or sensor_data.get('vibration_level', 0)
            
            # Historical averages
            avg_rop = sum([h.get('rop', 0) for h in history]) / len(history)
            avg_torque = sum([h.get('torque', 0) for h in history]) / len(history)
            avg_hook_load = sum([h.get('hook_load', 0) for h in history if h.get('hook_load')]) / max(1, len([h for h in history if h.get('hook_load')]))
            
            # Calculate changes
            rop_decrease = (avg_rop - current_rop) / max(avg_rop, 0.1) if avg_rop > 0 else 0
            torque_increase = (current_torque - avg_torque) / max(avg_torque, 0.1) if avg_torque > 0 else 0
            hook_load_decrease = (avg_hook_load - current_hook_load) / max(avg_hook_load, 0.1) if avg_hook_load > 0 and current_hook_load else 0
            
            # Indicators
            indicators = {
                "rop_decrease_percent": rop_decrease * 100,
                "torque_increase_percent": torque_increase * 100,
                "hook_load_decrease_percent": hook_load_decrease * 100,
                "current_rop": current_rop,
                "average_rop": avg_rop,
                "current_torque": current_torque,
                "average_torque": avg_torque,
                "vibration_level": current_vibration
            }
            
            # Risk calculation
            risk_score = 0.0
            risk_factors = []
            
            if rop_decrease > self.STUCK_PIPE_ROP_DECREASE_THRESHOLD:
                risk_score += 0.3
                risk_factors.append("Significant ROP decrease")
            
            if torque_increase > self.STUCK_PIPE_TORQUE_INCREASE_THRESHOLD:
                risk_score += 0.3
                risk_factors.append("Significant torque increase")
            
            if hook_load_decrease > self.STUCK_PIPE_HOOK_LOAD_DECREASE_THRESHOLD:
                risk_score += 0.2
                risk_factors.append("Hook load decrease")
            
            if current_vibration > 5.0:  # High vibration
                risk_score += 0.2
                risk_factors.append("High vibration level")
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "critical"
            elif risk_score >= 0.5:
                risk_level = "high"
            elif risk_score >= 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            stuck_pipe_detected = risk_score >= 0.5
            
            if stuck_pipe_detected:
                # Recommended actions based on risk level
                if risk_level == "critical":
                    recommended_actions = [
                        "STOP DRILLING IMMEDIATELY",
                        "Do not apply additional weight",
                        "Attempt to free pipe by rotating and reciprocating",
                        "Consider reducing mud weight if overbalanced",
                        "Notify supervisor immediately",
                        "Prepare for fishing operations if pipe cannot be freed"
                    ]
                elif risk_level == "high":
                    recommended_actions = [
                        "Reduce WOB immediately",
                        "Increase rotation speed gradually",
                        "Monitor torque and hook load closely",
                        "Consider back-reaming",
                        "Notify supervisor"
                    ]
                else:
                    recommended_actions = [
                        "Monitor parameters closely",
                        "Reduce WOB slightly",
                        "Consider adjusting drilling parameters",
                        "Notify supervisor if condition worsens"
                    ]
                
                # Create safety event
                event_id = self._create_safety_event(
                    rig_id=rig_id,
                    event_type=SafetyEventType.STUCK_PIPE.value,
                    severity=risk_level,
                    sensor_data_snapshot=sensor_data,
                    indicators=indicators,
                    recommendations=recommended_actions,
                    metadata={"risk_score": risk_score, "risk_factors": risk_factors}
                )
                
                # Broadcast alert
                try:
                    if websocket_manager:
                        alert_message = {
                            "message_type": "safety_alert",
                            "data": {
                                "event_type": "stuck_pipe",
                                "rig_id": rig_id,
                                "risk_level": risk_level,
                                "risk_score": risk_score,
                                "indicators": indicators,
                                "recommended_actions": recommended_actions,
                                "timestamp": datetime.now().isoformat()
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        websocket_manager.send_to_rig(rig_id, alert_message)
                except Exception as e:
                    logger.error(f"Error broadcasting stuck pipe alert: {e}")
                
                logger.warning(f"⚠️ STUCK PIPE DETECTED for rig {rig_id}: risk_level={risk_level}, risk_score={risk_score:.2f}")
                
                return {
                    "stuck_pipe_detected": True,
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "indicators": indicators,
                    "recommended_actions": recommended_actions,
                    "event_id": event_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "stuck_pipe_detected": False,
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "indicators": indicators,
                    "recommended_actions": [],
                    "event_id": None,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error detecting stuck pipe: {e}")
            return {
                "stuck_pipe_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_current_sensor_data(self, rig_id: str) -> Optional[Dict[str, Any]]:
        """Get current sensor data snapshot"""
        try:
            from services.data_service import DataService
            data_service = DataService()
            latest = data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
            if latest and len(latest) > 0:
                return latest[0]
            return None
        except Exception as e:
            logger.error(f"Error getting current sensor data: {e}")
            return None
    
    def _get_recent_sensor_data(self, rig_id: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent sensor data for comparison"""
        try:
            from services.data_service import DataService
            data_service = DataService()
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            history = data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            return history if history else []
        except Exception as e:
            logger.error(f"Error getting recent sensor data: {e}")
            return []
    
    def _calculate_pit_volume_change(self, current: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        """Calculate pit volume change (if pit volume data available)"""
        # This is a placeholder - actual implementation would use pit volume sensors
        # For now, estimate based on flow differential
        if not history:
            return 0.0
        
        current_flow_out = current.get('flow_out', 0)
        avg_flow_out = sum([h.get('flow_out', 0) for h in history]) / len(history) if history else 0
        
        # Estimate volume change (simplified)
        if current_flow_out and avg_flow_out:
            flow_diff = current_flow_out - avg_flow_out
            # Rough estimate: 1 gpm difference ≈ 0.1 bbl over 5 minutes
            return abs(flow_diff) * 0.1 * 5  # 5 minutes window
        return 0.0
    
    def _create_safety_event(
        self,
        rig_id: str,
        event_type: str,
        severity: str,
        sensor_data_snapshot: Dict[str, Any],
        indicators: Dict[str, Any],
        recommendations: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Create safety event record in database"""
        try:
            with db_manager.session_scope() as session:
                event = SafetyEventDB(
                    rig_id=rig_id,
                    event_type=event_type,
                    severity=severity,
                    status="active",
                    sensor_data_snapshot=sensor_data_snapshot,
                    recommendations=recommendations,
                    indicators=indicators,
                    metadata=metadata or {}
                )
                session.add(event)
                session.commit()
                session.refresh(event)
                return event.id
        except Exception as e:
            logger.error(f"Error creating safety event: {e}")
            return None


# Global singleton instance
safety_service = SafetyService()

