"""
Performance Metrics Service
Calculates real-time performance metrics for drilling operations
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from services.data_service import DataService

logger = logging.getLogger(__name__)


class PerformanceMetricsService:
    """Service for calculating drilling performance metrics"""
    
    def __init__(self):
        """Initialize PerformanceMetricsService"""
        self.data_service = DataService()
        
        # Cost constants (should come from configuration)
        self.RIG_TIME_COST_PER_HOUR = 5000.0  # USD per hour
        self.MUD_COST_PER_BARREL = 50.0  # USD per barrel
        self.BIT_COST = 50000.0  # USD per bit
        self.ENERGY_COST_PER_KWH = 0.15  # USD per kWh
        self.POWER_CONSUMPTION_KW = 2000.0  # Average power consumption in kW
    
    def calculate_real_time_metrics(self, rig_id: str, session_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate real-time performance metrics
        
        Args:
            rig_id: Rig identifier
            session_id: Optional drilling session ID
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get latest sensor data
            latest_data = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
            if not latest_data or len(latest_data) == 0:
                return {"error": "No sensor data available"}
            
            current = latest_data[0]
            
            # Get session data if available
            session_data = None
            if session_id:
                session_data = self._get_session_data(session_id)
            
            # Calculate metrics
            metrics = {
                "rop_efficiency": self._calculate_rop_efficiency(current),
                "energy_efficiency": self._calculate_energy_efficiency(current),
                "bit_life_remaining": self._calculate_bit_life_remaining(current),
                "drilling_efficiency_index": self._calculate_dei(current),
            }
            
            # Calculate cost metrics if session data available
            if session_data:
                metrics.update({
                    "cost_per_meter": self._calculate_cost_per_meter(session_data, current),
                    "total_cost": self._calculate_total_cost(session_data),
                    "projected_total_cost": self._project_total_cost(session_data, current)
                })
            
            # Calculate time estimates
            if session_data and session_data.get('target_depth'):
                metrics["estimated_time_to_target"] = self._estimate_time_to_target(
                    current,
                    session_data.get('target_depth'),
                    session_data.get('start_depth', 0)
                )
            
            metrics["timestamp"] = datetime.now().isoformat()
            metrics["rig_id"] = rig_id
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_rop_efficiency(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate ROP efficiency (ROP / Energy Consumption)"""
        rop = sensor_data.get('rop', 0)
        wob = sensor_data.get('wob', 0)
        rpm = sensor_data.get('rpm', 0)
        
        if rop <= 0:
            return 0.0
        
        # Energy consumption estimate (simplified)
        # Energy ≈ WOB × RPM × constant
        energy_consumption = (wob * rpm) / 1000.0 if wob > 0 and rpm > 0 else 1.0
        
        # Efficiency = ROP / Energy (normalized)
        efficiency = (rop / max(energy_consumption, 0.1)) * 100.0
        
        return min(efficiency, 100.0)  # Cap at 100%
    
    def _calculate_energy_efficiency(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate energy efficiency"""
        rop = sensor_data.get('rop', 0)
        power_consumption = sensor_data.get('power_consumption', self.POWER_CONSUMPTION_KW)
        
        if rop <= 0 or power_consumption <= 0:
            return 0.0
        
        # Energy efficiency = ROP / Power (normalized)
        efficiency = (rop / power_consumption) * 1000.0  # Normalize
        
        return min(efficiency, 100.0)
    
    def _calculate_bit_life_remaining(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate estimated bit life remaining (percentage)"""
        # This would typically use bit wear models
        # For now, use a simplified calculation based on vibration and torque
        vibration = sensor_data.get('vibration', 0) or sensor_data.get('vibration_level', 0)
        torque = sensor_data.get('torque', 0)
        
        # Simplified: higher vibration and torque = more wear
        wear_factor = min(1.0, (vibration / 10.0) * 0.5 + (torque / 100000.0) * 0.5)
        life_remaining = max(0.0, (1.0 - wear_factor) * 100.0)
        
        return life_remaining
    
    def _calculate_dei(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate Drilling Efficiency Index (0-100)"""
        rop = sensor_data.get('rop', 0)
        wob = sensor_data.get('wob', 0)
        rpm = sensor_data.get('rpm', 0)
        vibration = sensor_data.get('vibration', 0) or sensor_data.get('vibration_level', 0)
        
        if rop <= 0:
            return 0.0
        
        # Normalize parameters
        rop_score = min(100.0, (rop / 100.0) * 100.0)  # Assume 100 ft/hr is excellent
        wob_score = min(100.0, (wob / 50000.0) * 100.0) if wob > 0 else 0
        rpm_score = min(100.0, (rpm / 150.0) * 100.0) if rpm > 0 else 0
        
        # Penalty for high vibration
        vibration_penalty = min(30.0, vibration * 3.0)
        
        # Weighted average
        dei = (rop_score * 0.5 + wob_score * 0.2 + rpm_score * 0.2) - vibration_penalty
        
        return max(0.0, min(100.0, dei))
    
    def _calculate_cost_per_meter(self, session_data: Dict[str, Any], current_data: Dict[str, Any]) -> float:
        """Calculate cost per meter drilled"""
        total_cost = self._calculate_total_cost(session_data)
        depth_drilled = current_data.get('depth', 0) - session_data.get('start_depth', 0)
        
        if depth_drilled <= 0:
            return 0.0
        
        return total_cost / depth_drilled
    
    def _calculate_total_cost(self, session_data: Dict[str, Any]) -> float:
        """Calculate total cost for drilling session"""
        # Rig time cost
        start_time = session_data.get('start_time')
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        elif not isinstance(start_time, datetime):
            start_time = datetime.now()
        
        hours_elapsed = (datetime.now() - start_time).total_seconds() / 3600.0
        rig_time_cost = hours_elapsed * self.RIG_TIME_COST_PER_HOUR
        
        # Mud cost (simplified - would need mud volume tracking)
        mud_cost = 0.0  # Placeholder
        
        # Bit cost (amortized)
        bit_cost = self.BIT_COST  # Full cost for now
        
        # Energy cost
        energy_cost = hours_elapsed * self.POWER_CONSUMPTION_KW * self.ENERGY_COST_PER_KWH
        
        total_cost = rig_time_cost + mud_cost + bit_cost + energy_cost
        
        return total_cost
    
    def _project_total_cost(self, session_data: Dict[str, Any], current_data: Dict[str, Any]) -> float:
        """Project total cost to target depth"""
        current_cost = self._calculate_total_cost(session_data)
        current_depth = current_data.get('depth', 0)
        start_depth = session_data.get('start_depth', 0)
        target_depth = session_data.get('target_depth', 0)
        
        if target_depth <= current_depth or current_depth <= start_depth:
            return current_cost
        
        depth_drilled = current_depth - start_depth
        depth_remaining = target_depth - current_depth
        
        if depth_drilled <= 0:
            return current_cost
        
        cost_per_meter = current_cost / depth_drilled
        projected_cost = current_cost + (cost_per_meter * depth_remaining)
        
        return projected_cost
    
    def _estimate_time_to_target(self, current_data: Dict[str, Any], target_depth: float, start_depth: float) -> float:
        """Estimate time to reach target depth (hours)"""
        current_depth = current_data.get('depth', 0)
        rop = current_data.get('rop', 0)
        
        if rop <= 0 or target_depth <= current_depth:
            return 0.0
        
        depth_remaining = target_depth - current_depth
        hours_remaining = depth_remaining / rop
        
        return hours_remaining
    
    def _get_session_data(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get drilling session data"""
        try:
            from api.models.database_models import DrillingSessionDB
            with db_manager.session_scope() as session:
                drilling_session = session.query(DrillingSessionDB).filter(
                    DrillingSessionDB.id == session_id
                ).first()
                
                if drilling_session:
                    return {
                        "start_time": drilling_session.start_time,
                        "start_depth": drilling_session.start_depth,
                        "target_depth": getattr(drilling_session, 'target_depth', None),
                        "average_rop": drilling_session.average_rop
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return None


# Global singleton instance
performance_metrics_service = PerformanceMetricsService()

