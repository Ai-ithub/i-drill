"""
Real-time Optimization Service
Integration of RL models with live data, real-time performance metrics, cost tracking, and optimization recommendations
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import os

from services.rl_service import rl_service
from services.integration_service import integration_service
from services.data_service import DataService
from services.control_service import control_service
from services.websocket_manager import websocket_manager
from services.performance_metrics_service import performance_metrics_service

logger = logging.getLogger(__name__)


class RealTimeOptimizationService:
    """
    Real-time Optimization Service.
    
    Features:
    - Integration of RL models (PPO/SAC) with live data
    - Real-time inference for parameter recommendations
    - Automatic or suggested application to driller
    - Real-time performance metrics (ROP Efficiency, Energy Efficiency, DEI)
    - Real-time cost tracking
    - Optimization recommendations
    """
    
    def __init__(self):
        """Initialize RealTimeOptimizationService."""
        self.data_service = DataService()
        self.running = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # Check every 5 seconds
        
        # RL model configuration
        self.rl_models: Dict[str, Dict[str, Any]] = {}  # rig_id -> model info
        self.auto_apply_enabled = False  # Whether to auto-apply RL recommendations
        self.auto_apply_rigs: set = set()  # Rigs with auto-apply enabled
        
        # Performance metrics cache
        self.performance_cache: Dict[str, Dict[str, Any]] = {}  # rig_id -> metrics
        
        # Cost tracking
        self.cost_tracking: Dict[str, Dict[str, Any]] = {}  # rig_id -> cost data
        self.cost_per_meter_history: Dict[str, deque] = {}  # rig_id -> deque of costs
        
        # Recommendations cache
        self.recommendations_cache: Dict[str, List[Dict[str, Any]]] = {}  # rig_id -> recommendations
        
        logger.info("Real-time optimization service initialized")
    
    def load_rl_model(
        self,
        rig_id: str,
        model_path: str,
        model_type: str = "PPO",  # PPO or SAC
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Load RL model for a rig.
        
        Args:
            rig_id: Rig identifier
            model_path: Path to trained model
            model_type: Model type (PPO or SAC)
            auto_apply: Whether to auto-apply recommendations
        
        Returns:
            Model loading result
        """
        try:
            # Load model using RL service
            # Note: This assumes model is compatible with RL service's policy wrapper
            result = rl_service.attach_policy(
                source="file",
                identifier=model_path,
                stage="production"
            )
            
            if result.get("success"):
                self.rl_models[rig_id] = {
                    "model_path": model_path,
                    "model_type": model_type,
                    "loaded_at": datetime.now().isoformat(),
                    "auto_apply": auto_apply
                }
                
                if auto_apply:
                    self.auto_apply_rigs.add(rig_id)
                
                logger.info(f"RL model loaded for rig {rig_id}: {model_type} from {model_path}")
                
                return {
                    "success": True,
                    "message": f"RL model ({model_type}) loaded successfully",
                    "rig_id": rig_id,
                    "model_type": model_type,
                    "auto_apply": auto_apply
                }
            else:
                return {
                    "success": False,
                    "message": result.get("message", "Failed to load model"),
                    "rig_id": rig_id
                }
        
        except Exception as e:
            logger.error(f"Error loading RL model for rig {rig_id}: {e}")
            return {
                "success": False,
                "message": str(e),
                "rig_id": rig_id
            }
    
    def get_realtime_recommendation(
        self,
        rig_id: str,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get real-time parameter recommendation from RL model.
        
        Args:
            rig_id: Rig identifier
            sensor_data: Optional current sensor data (if None, fetches latest)
        
        Returns:
            Recommendation with suggested parameters
        """
        try:
            if rig_id not in self.rl_models:
                return {
                    "success": False,
                    "message": f"No RL model loaded for rig {rig_id}",
                    "recommendation": None
                }
            
            # Get latest sensor data if not provided
            if sensor_data is None:
                latest = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
                if not latest or len(latest) == 0:
                    return {
                        "success": False,
                        "message": "No sensor data available",
                        "recommendation": None
                    }
                sensor_data = latest[0]
            
            # Convert sensor data to RL observation
            observation = integration_service.sensor_to_rl_observation(sensor_data)
            
            if not observation:
                return {
                    "success": False,
                    "message": "Failed to convert sensor data to RL observation",
                    "recommendation": None
                }
            
            # Get action from RL model
            try:
                # Use RL service's auto_step to get recommendation
                rl_state = rl_service.auto_step()
                
                if rl_state and "action" in rl_state:
                    action = rl_state["action"]
                    
                    recommendation = {
                        "wob": action.get("wob"),
                        "rpm": action.get("rpm"),
                        "flow_rate": action.get("flow_rate"),
                        "confidence": 0.85,  # Could be calculated from model uncertainty
                        "model_type": self.rl_models[rig_id]["model_type"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Check if auto-apply is enabled
                    auto_apply = self.rl_models[rig_id].get("auto_apply", False)
                    
                    if auto_apply:
                        # Apply recommendation automatically
                        apply_result = self._apply_recommendation(rig_id, recommendation)
                        recommendation["applied"] = True
                        recommendation["apply_result"] = apply_result
                    else:
                        recommendation["applied"] = False
                    
                    return {
                        "success": True,
                        "message": "Recommendation generated",
                        "recommendation": recommendation,
                        "auto_apply": auto_apply
                    }
                else:
                    return {
                        "success": False,
                        "message": "RL model did not return action",
                        "recommendation": None
                    }
            
            except Exception as e:
                logger.error(f"Error getting RL recommendation: {e}")
                return {
                    "success": False,
                    "message": str(e),
                    "recommendation": None
                }
        
        except Exception as e:
            logger.error(f"Error in get_realtime_recommendation: {e}")
            return {
                "success": False,
                "message": str(e),
                "recommendation": None
            }
    
    def _apply_recommendation(
        self,
        rig_id: str,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply recommendation to control system."""
        try:
            actions_taken = []
            
            # Apply WOB
            if recommendation.get("wob") is not None:
                result = control_service.apply_parameter_change(
                    rig_id=rig_id,
                    component="drilling",
                    parameter="wob",
                    new_value=recommendation["wob"],
                    metadata={
                        "source": "rl_optimization",
                        "model_type": recommendation.get("model_type"),
                        "confidence": recommendation.get("confidence")
                    }
                )
                if result.get("success"):
                    actions_taken.append("WOB updated")
            
            # Apply RPM
            if recommendation.get("rpm") is not None:
                result = control_service.apply_parameter_change(
                    rig_id=rig_id,
                    component="drilling",
                    parameter="rpm",
                    new_value=recommendation["rpm"],
                    metadata={
                        "source": "rl_optimization",
                        "model_type": recommendation.get("model_type")
                    }
                )
                if result.get("success"):
                    actions_taken.append("RPM updated")
            
            # Apply Mud Flow
            if recommendation.get("flow_rate") is not None:
                result = control_service.apply_parameter_change(
                    rig_id=rig_id,
                    component="mud_system",
                    parameter="mud_flow",
                    new_value=recommendation["flow_rate"],
                    metadata={
                        "source": "rl_optimization",
                        "model_type": recommendation.get("model_type")
                    }
                )
                if result.get("success"):
                    actions_taken.append("Mud flow updated")
            
            return {
                "success": len(actions_taken) > 0,
                "actions_taken": actions_taken,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error applying recommendation: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions_taken": []
            }
    
    def calculate_realtime_performance_metrics(
        self,
        rig_id: str,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate real-time performance metrics.
        
        Metrics:
        - ROP Efficiency
        - Energy Efficiency
        - Drilling Efficiency Index (DEI)
        
        Args:
            rig_id: Rig identifier
            sensor_data: Optional current sensor data
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # Get latest sensor data if not provided
            if sensor_data is None:
                latest = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
                if not latest or len(latest) == 0:
                    return {
                        "success": False,
                        "message": "No sensor data available",
                        "metrics": None
                    }
                sensor_data = latest[0]
            
            # Get historical data for comparison
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=10)
            history = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            # Current values
            current_rop = sensor_data.get("rop") or sensor_data.get("ROP", 0)
            current_wob = sensor_data.get("wob") or sensor_data.get("WOB", 0)
            current_rpm = sensor_data.get("rpm") or sensor_data.get("RPM", 0)
            current_torque = sensor_data.get("torque") or sensor_data.get("Torque", 0)
            current_power = sensor_data.get("power_consumption") or sensor_data.get("Power_Consumption", 0)
            
            # Calculate ROP Efficiency
            # ROP Efficiency = (Actual ROP / Theoretical Max ROP) * 100
            theoretical_max_rop = self._calculate_theoretical_max_rop(
                wob=current_wob,
                rpm=current_rpm,
                torque=current_torque
            )
            rop_efficiency = (current_rop / theoretical_max_rop * 100) if theoretical_max_rop > 0 else 0
            
            # Calculate Energy Efficiency
            # Energy Efficiency = (ROP / Power Consumption) * 1000 (m/kWh)
            energy_efficiency = (current_rop / current_power * 1000) if current_power > 0 else 0
            
            # Calculate Drilling Efficiency Index (DEI)
            # DEI = (ROP * WOB) / (Torque * RPM) * 100
            dei = ((current_rop * current_wob) / (current_torque * current_rpm * 0.01)) if (current_torque * current_rpm) > 0 else 0
            
            # Historical averages for comparison
            if history:
                avg_rop = sum([h.get("rop", 0) or h.get("ROP", 0) for h in history]) / len(history)
                avg_energy_efficiency = sum([
                    ((h.get("rop", 0) or h.get("ROP", 0)) / (h.get("power_consumption", 1) or h.get("Power_Consumption", 1)) * 1000)
                    for h in history if (h.get("power_consumption") or h.get("Power_Consumption"))
                ]) / max(1, len([h for h in history if (h.get("power_consumption") or h.get("Power_Consumption"))]))
            else:
                avg_rop = current_rop
                avg_energy_efficiency = energy_efficiency
            
            metrics = {
                "rop_efficiency": {
                    "current": round(rop_efficiency, 2),
                    "unit": "%",
                    "description": "Rate of Penetration Efficiency"
                },
                "energy_efficiency": {
                    "current": round(energy_efficiency, 2),
                    "average": round(avg_energy_efficiency, 2),
                    "unit": "m/kWh",
                    "description": "Energy Efficiency"
                },
                "drilling_efficiency_index": {
                    "current": round(dei, 2),
                    "unit": "index",
                    "description": "Drilling Efficiency Index"
                },
                "current_rop": current_rop,
                "theoretical_max_rop": theoretical_max_rop,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache metrics
            self.performance_cache[rig_id] = metrics
            
            return {
                "success": True,
                "metrics": metrics,
                "rig_id": rig_id
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "success": False,
                "message": str(e),
                "metrics": None
            }
    
    def _calculate_theoretical_max_rop(
        self,
        wob: float,
        rpm: float,
        torque: float
    ) -> float:
        """Calculate theoretical maximum ROP based on parameters."""
        # Simplified theoretical ROP calculation
        # In practice, this would use more sophisticated drilling models
        if wob <= 0 or rpm <= 0:
            return 0.0
        
        # Basic formula: ROP ∝ (WOB * RPM) / Torque
        theoretical_rop = (wob * rpm) / max(torque, 1.0) * 0.1  # Scaling factor
        return max(theoretical_rop, 0.0)
    
    def calculate_realtime_cost(
        self,
        rig_id: str,
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate real-time cost tracking.
        
        Metrics:
        - Cost per Meter
        - Total Cost
        - Comparison with budget
        
        Args:
            rig_id: Rig identifier
            sensor_data: Optional current sensor data
        
        Returns:
            Cost tracking dictionary
        """
        try:
            # Get latest sensor data if not provided
            if sensor_data is None:
                latest = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
                if not latest or len(latest) == 0:
                    return {
                        "success": False,
                        "message": "No sensor data available",
                        "cost": None
                    }
                sensor_data = latest[0]
            
            # Get cost configuration (could be from database or config)
            cost_config = self._get_cost_configuration(rig_id)
            
            current_depth = sensor_data.get("depth") or sensor_data.get("Depth", 0)
            current_power = sensor_data.get("power_consumption") or sensor_data.get("Power_Consumption", 0)
            
            # Get historical data for total cost calculation
            start_time = datetime.now() - timedelta(days=30)  # Last 30 days
            history = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=datetime.now(),
                limit=10000
            )
            
            # Calculate time-based costs
            time_cost = len(history) * cost_config.get("cost_per_minute", 10.0)  # $10 per minute
            
            # Calculate energy costs
            total_energy = sum([
                (h.get("power_consumption", 0) or h.get("Power_Consumption", 0)) / 60.0  # kWh per minute
                for h in history
            ])
            energy_cost = total_energy * cost_config.get("energy_cost_per_kwh", 0.15)  # $0.15 per kWh
            
            # Calculate total depth drilled
            if history:
                initial_depth = history[0].get("depth") or history[0].get("Depth", 0)
                total_depth_drilled = current_depth - initial_depth
            else:
                total_depth_drilled = 0
            
            # Calculate cost per meter
            cost_per_meter = (time_cost + energy_cost) / max(total_depth_drilled, 0.1)
            
            # Total cost
            total_cost = time_cost + energy_cost
            
            # Budget comparison
            budget = cost_config.get("budget", 1000000.0)  # $1M default budget
            budget_utilization = (total_cost / budget * 100) if budget > 0 else 0
            remaining_budget = budget - total_cost
            
            # Projected total cost (if current rate continues)
            if total_depth_drilled > 0:
                target_depth = cost_config.get("target_depth", 5000.0)  # 5000m target
                remaining_depth = target_depth - current_depth
                projected_total_cost = total_cost + (cost_per_meter * remaining_depth)
            else:
                projected_total_cost = total_cost
            
            cost_data = {
                "cost_per_meter": round(cost_per_meter, 2),
                "total_cost": round(total_cost, 2),
                "projected_total_cost": round(projected_total_cost, 2),
                "budget": budget,
                "budget_utilization": round(budget_utilization, 2),
                "remaining_budget": round(remaining_budget, 2),
                "total_depth_drilled": round(total_depth_drilled, 2),
                "current_depth": current_depth,
                "time_cost": round(time_cost, 2),
                "energy_cost": round(energy_cost, 2),
                "currency": "USD",
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cost tracking
            self.cost_tracking[rig_id] = cost_data
            
            # Update history
            if rig_id not in self.cost_per_meter_history:
                self.cost_per_meter_history[rig_id] = deque(maxlen=1000)
            self.cost_per_meter_history[rig_id].append(cost_per_meter)
            
            return {
                "success": True,
                "cost": cost_data,
                "rig_id": rig_id
            }
        
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return {
                "success": False,
                "message": str(e),
                "cost": None
            }
    
    def _get_cost_configuration(self, rig_id: str) -> Dict[str, Any]:
        """Get cost configuration for a rig."""
        # In production, this would come from database or config file
        return {
            "cost_per_minute": 10.0,  # $10 per minute of drilling
            "energy_cost_per_kwh": 0.15,  # $0.15 per kWh
            "budget": 1000000.0,  # $1M budget
            "target_depth": 5000.0  # 5000m target depth
        }
    
    def generate_optimization_recommendations(
        self,
        rig_id: str
    ) -> Dict[str, Any]:
        """
        Generate optimization recommendations.
        
        Recommendations for:
        - Improving ROP
        - Reducing cost
        - Reducing energy consumption
        
        Args:
            rig_id: Rig identifier
        
        Returns:
            Recommendations dictionary
        """
        try:
            recommendations = []
            
            # Get current metrics
            metrics_result = self.calculate_realtime_performance_metrics(rig_id)
            cost_result = self.calculate_realtime_cost(rig_id)
            
            if not metrics_result.get("success") or not cost_result.get("success"):
                return {
                    "success": False,
                    "message": "Failed to get metrics or cost data",
                    "recommendations": []
                }
            
            metrics = metrics_result.get("metrics", {})
            cost = cost_result.get("cost", {})
            
            # ROP Efficiency recommendations
            rop_efficiency = metrics.get("rop_efficiency", {}).get("current", 0)
            if rop_efficiency < 70:
                recommendations.append({
                    "type": "improve_rop",
                    "priority": "high",
                    "title": "Improve ROP Efficiency",
                    "description": f"Current ROP efficiency is {rop_efficiency:.1f}%. Consider optimizing drilling parameters.",
                    "suggestions": [
                        "Increase WOB gradually if torque allows",
                        "Optimize RPM for current formation",
                        "Check bit condition and consider replacement if worn"
                    ],
                    "expected_improvement": "10-15% ROP increase"
                })
            
            # Energy Efficiency recommendations
            energy_efficiency = metrics.get("energy_efficiency", {}).get("current", 0)
            avg_energy_efficiency = metrics.get("energy_efficiency", {}).get("average", 0)
            if energy_efficiency < avg_energy_efficiency * 0.9:
                recommendations.append({
                    "type": "reduce_energy",
                    "priority": "medium",
                    "title": "Reduce Energy Consumption",
                    "description": f"Energy efficiency is below average. Current: {energy_efficiency:.2f} m/kWh, Average: {avg_energy_efficiency:.2f} m/kWh",
                    "suggestions": [
                        "Optimize RPM to reduce power consumption",
                        "Consider reducing mud flow if well conditions allow",
                        "Check for equipment inefficiencies"
                    ],
                    "expected_improvement": "5-10% energy reduction"
                })
            
            # Cost recommendations
            cost_per_meter = cost.get("cost_per_meter", 0)
            budget_utilization = cost.get("budget_utilization", 0)
            if budget_utilization > 80:
                recommendations.append({
                    "type": "reduce_cost",
                    "priority": "high",
                    "title": "Budget Alert",
                    "description": f"Budget utilization is {budget_utilization:.1f}%. Cost per meter: ${cost_per_meter:.2f}",
                    "suggestions": [
                        "Focus on improving ROP to reduce time-based costs",
                        "Optimize energy consumption",
                        "Review and optimize non-productive time"
                    ],
                    "expected_improvement": "10-20% cost reduction"
                })
            elif cost_per_meter > 500:  # High cost per meter
                recommendations.append({
                    "type": "reduce_cost",
                    "priority": "medium",
                    "title": "High Cost per Meter",
                    "description": f"Cost per meter is ${cost_per_meter:.2f}, which is above optimal range.",
                    "suggestions": [
                        "Improve drilling efficiency",
                        "Reduce non-productive time",
                        "Optimize equipment utilization"
                    ],
                    "expected_improvement": "15-25% cost reduction"
                })
            
            # RL-based recommendations
            if rig_id in self.rl_models:
                rl_recommendation = self.get_realtime_recommendation(rig_id)
                if rl_recommendation.get("success") and rl_recommendation.get("recommendation"):
                    rec = rl_recommendation["recommendation"]
                    recommendations.append({
                        "type": "rl_optimization",
                        "priority": "high",
                        "title": "RL Model Recommendation",
                        "description": f"RL model ({self.rl_models[rig_id]['model_type']}) suggests optimal parameters",
                        "suggestions": [
                            f"WOB: {rec.get('wob', 'N/A')} tons",
                            f"RPM: {rec.get('rpm', 'N/A')}",
                            f"Mud Flow: {rec.get('flow_rate', 'N/A')} L/min"
                        ],
                        "confidence": rec.get("confidence", 0),
                        "model_type": rec.get("model_type"),
                        "applied": rec.get("applied", False)
                    })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
            
            # Cache recommendations
            self.recommendations_cache[rig_id] = recommendations
            
            return {
                "success": True,
                "recommendations": recommendations,
                "count": len(recommendations),
                "rig_id": rig_id,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "success": False,
                "message": str(e),
                "recommendations": []
            }
    
    def start_monitoring(self) -> None:
        """Start real-time optimization monitoring."""
        if self.running:
            logger.warning("Real-time optimization monitoring is already running")
            return
        
        self.running = True
        import threading
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="RealTimeOptimization-Monitoring"
        )
        self.monitoring_thread.start()
        logger.info("Real-time optimization monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time optimization monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Real-time optimization monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        import time
        
        while self.running:
            try:
                # Get active rigs
                rigs = self._get_active_rigs()
                
                for rig_id in rigs:
                    try:
                        # Calculate performance metrics
                        self.calculate_realtime_performance_metrics(rig_id)
                        
                        # Calculate cost
                        self.calculate_realtime_cost(rig_id)
                        
                        # Generate recommendations
                        self.generate_optimization_recommendations(rig_id)
                        
                        # Get RL recommendations if model loaded
                        if rig_id in self.rl_models:
                            self.get_realtime_recommendation(rig_id)
                    
                    except Exception as e:
                        logger.error(f"Error monitoring rig {rig_id}: {e}")
                
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Error in optimization monitoring loop: {e}")
                time.sleep(1)
    
    def _get_active_rigs(self) -> List[str]:
        """Get list of active rig IDs."""
        try:
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


# Global instance
realtime_optimization_service = RealTimeOptimizationService()

