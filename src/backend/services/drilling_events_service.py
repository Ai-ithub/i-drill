"""
Drilling Events Service
Handles formation change detection and other drilling event detection
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from api.models.database_models import DrillingEventDB
from database import db_manager
from services.data_service import DataService

logger = logging.getLogger(__name__)


class DrillingEventsService:
    """Service for drilling event detection"""
    
    def __init__(self):
        """Initialize DrillingEventsService"""
        self.data_service = DataService()
        
        # Formation change detection thresholds
        self.FORMATION_CHANGE_GAMMA_RAY_THRESHOLD = 20.0  # API units
        self.FORMATION_CHANGE_RESISTIVITY_THRESHOLD = 2.0  # ohm-m
        self.FORMATION_CHANGE_ROP_PATTERN_THRESHOLD = 0.3  # 30% change
    
    def detect_formation_change(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect formation change based on sensor data
        
        Args:
            sensor_data: Current sensor data dictionary
            
        Returns:
            Dictionary with detection result
        """
        try:
            rig_id = sensor_data.get('rig_id')
            if not rig_id:
                return {"formation_change_detected": False, "error": "rig_id missing"}
            
            # Get historical data for comparison
            history = self._get_recent_sensor_data(rig_id, minutes=10)
            if not history or len(history) < 5:
                return {"formation_change_detected": False, "error": "insufficient history"}
            
            # Current values
            current_gamma_ray = sensor_data.get('gamma_ray')
            current_resistivity = sensor_data.get('resistivity')
            current_rop = sensor_data.get('rop', 0)
            current_depth = sensor_data.get('depth', 0)
            
            # Historical averages
            gamma_ray_history = [h.get('gamma_ray') for h in history if h.get('gamma_ray')]
            resistivity_history = [h.get('resistivity') for h in history if h.get('resistivity')]
            rop_history = [h.get('rop', 0) for h in history]
            
            avg_gamma_ray = sum(gamma_ray_history) / len(gamma_ray_history) if gamma_ray_history else None
            avg_resistivity = sum(resistivity_history) / len(resistivity_history) if resistivity_history else None
            avg_rop = sum(rop_history) / len(rop_history) if rop_history else 0
            
            # Calculate changes
            indicators = {}
            change_score = 0.0
            detected = False
            
            if current_gamma_ray and avg_gamma_ray:
                gamma_ray_change = abs(current_gamma_ray - avg_gamma_ray)
                indicators["gamma_ray_change"] = gamma_ray_change
                indicators["gamma_ray_current"] = current_gamma_ray
                indicators["gamma_ray_average"] = avg_gamma_ray
                if gamma_ray_change > self.FORMATION_CHANGE_GAMMA_RAY_THRESHOLD:
                    change_score += 0.4
                    detected = True
            
            if current_resistivity and avg_resistivity:
                resistivity_change = abs(current_resistivity - avg_resistivity)
                indicators["resistivity_change"] = resistivity_change
                indicators["resistivity_current"] = current_resistivity
                indicators["resistivity_average"] = avg_resistivity
                if resistivity_change > self.FORMATION_CHANGE_RESISTIVITY_THRESHOLD:
                    change_score += 0.4
                    detected = True
            
            if current_rop and avg_rop:
                rop_change = abs(current_rop - avg_rop) / max(avg_rop, 0.1)
                indicators["rop_change_percent"] = rop_change * 100
                indicators["rop_current"] = current_rop
                indicators["rop_average"] = avg_rop
                if rop_change > self.FORMATION_CHANGE_ROP_PATTERN_THRESHOLD:
                    change_score += 0.2
                    detected = True
            
            if detected:
                # Classify formation type based on gamma ray and resistivity
                formation_type = self._classify_formation(current_gamma_ray, current_resistivity)
                
                # Get recommended parameters for this formation
                recommended_parameters = self._get_recommended_parameters(formation_type, sensor_data)
                
                # Create drilling event
                event_id = self._create_drilling_event(
                    rig_id=rig_id,
                    event_type="formation_change",
                    severity="medium",
                    depth=current_depth,
                    sensor_data_snapshot=sensor_data,
                    metadata={
                        "formation_type": formation_type,
                        "indicators": indicators,
                        "change_score": change_score
                    }
                )
                
                confidence = min(0.95, change_score)
                
                logger.info(f"Formation change detected for rig {rig_id} at depth {current_depth}: {formation_type}")
                
                return {
                    "formation_change_detected": True,
                    "depth": current_depth,
                    "confidence": confidence,
                    "formation_type": formation_type,
                    "recommended_parameters": recommended_parameters,
                    "event_id": event_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "formation_change_detected": False,
                    "depth": current_depth,
                    "confidence": 0.0,
                    "formation_type": None,
                    "recommended_parameters": {},
                    "event_id": None,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error detecting formation change: {e}")
            return {
                "formation_change_detected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_recent_sensor_data(self, rig_id: str, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent sensor data for comparison"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            history = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            return history if history else []
        except Exception as e:
            logger.error(f"Error getting recent sensor data: {e}")
            return []
    
    def _classify_formation(self, gamma_ray: Optional[float], resistivity: Optional[float]) -> str:
        """Classify formation type based on gamma ray and resistivity"""
        # Simplified classification - in production, use more sophisticated models
        if gamma_ray is None and resistivity is None:
            return "unknown"
        
        if gamma_ray:
            if gamma_ray < 30:
                return "sandstone"
            elif gamma_ray < 60:
                return "limestone"
            elif gamma_ray < 100:
                return "shale"
            else:
                return "clay"
        
        if resistivity:
            if resistivity > 10:
                return "sandstone"
            elif resistivity > 5:
                return "limestone"
            else:
                return "shale"
        
        return "unknown"
    
    def _get_recommended_parameters(self, formation_type: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommended drilling parameters for formation type"""
        # Base recommendations - in production, use ML models or expert systems
        base_wob = current_data.get('wob', 0)
        base_rpm = current_data.get('rpm', 0)
        base_mud_flow = current_data.get('mud_flow', 0)
        
        recommendations = {
            "wob": base_wob,
            "rpm": base_rpm,
            "mud_flow": base_mud_flow
        }
        
        if formation_type == "sandstone":
            recommendations["wob"] = base_wob * 1.1  # Increase WOB
            recommendations["rpm"] = base_rpm * 0.9  # Slightly reduce RPM
        elif formation_type == "shale":
            recommendations["wob"] = base_wob * 0.9  # Reduce WOB
            recommendations["rpm"] = base_rpm * 1.1  # Increase RPM
        elif formation_type == "limestone":
            recommendations["wob"] = base_wob * 1.05
            recommendations["rpm"] = base_rpm * 1.0
        
        return recommendations
    
    def _create_drilling_event(
        self,
        rig_id: str,
        event_type: str,
        severity: str,
        depth: Optional[float],
        sensor_data_snapshot: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[int]:
        """Create drilling event record in database"""
        try:
            with db_manager.session_scope() as session:
                event = DrillingEventDB(
                    rig_id=rig_id,
                    event_type=event_type,
                    severity=severity,
                    depth=depth,
                    sensor_data_snapshot=sensor_data_snapshot,
                    metadata=metadata
                )
                session.add(event)
                session.commit()
                session.refresh(event)
                return event.id
        except Exception as e:
            logger.error(f"Error creating drilling event: {e}")
            return None


# Global singleton instance
drilling_events_service = DrillingEventsService()

