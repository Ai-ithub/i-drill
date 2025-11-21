"""
Predictive Maintenance Real-time Service
RUL Prediction, Anomaly Detection with Isolation Forest, and Maintenance Scheduling
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from services.prediction_service import PredictionService
from services.data_service import DataService
from services.alert_management_service import alert_management_service
from services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

# Try to import Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Isolation Forest anomaly detection will be disabled.")


class PredictiveMaintenanceService:
    """
    Real-time Predictive Maintenance Service.
    
    Features:
    - RUL Prediction Real-time using LSTM/Transformer models
    - Anomaly Detection Real-time using Isolation Forest
    - Maintenance Scheduling based on RUL and anomalies
    - Alerts for low RUL equipment
    """
    
    def __init__(self):
        """Initialize PredictiveMaintenanceService."""
        self.prediction_service = PredictionService()
        self.data_service = DataService()
        self.running = False
        self.monitoring_thread = None
        self.monitoring_interval = 30.0  # Check every 30 seconds
        
        # RUL thresholds for alerts (hours)
        self.rul_alert_thresholds = {
            "critical": 24,  # Less than 24 hours
            "high": 72,  # Less than 72 hours (3 days)
            "medium": 168,  # Less than 168 hours (1 week)
            "low": 720  # Less than 720 hours (30 days)
        }
        
        # Isolation Forest models for each component
        self.isolation_forest_models: Dict[str, IsolationForest] = {}
        self.anomaly_history: Dict[str, deque] = {}  # component -> deque of anomalies
        
        # Component definitions
        self.components = [
            "drill_bit",
            "mud_pump",
            "top_drive",
            "drawworks",
            "rotary_table",
            "bop",
            "mud_system",
            "power_system"
        ]
        
        # RUL cache
        self.rul_cache: Dict[str, Dict[str, Any]] = {}  # rig_id -> component -> RUL data
        
        logger.info("Predictive maintenance service initialized")
    
    def predict_rul_realtime(
        self,
        rig_id: str,
        component: str = "general",
        model_type: str = "lstm",
        lookback_window: int = 50,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Predict RUL in real-time for a component.
        
        Args:
            rig_id: Rig identifier
            component: Component name (e.g., "drill_bit", "mud_pump")
            model_type: Model type (lstm, transformer, cnn_lstm)
            lookback_window: Number of data points to use
            lookback_hours: Hours of historical data to retrieve
        
        Returns:
            RUL prediction result
        """
        try:
            # Get historical sensor data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            sensor_data = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=lookback_window * 2  # Get more than needed
            )
            
            if not sensor_data or len(sensor_data) < lookback_window:
                return {
                    "success": False,
                    "message": f"Insufficient data: need {lookback_window} points, got {len(sensor_data) if sensor_data else 0}",
                    "predicted_rul": None
                }
            
            # Use last lookback_window points
            recent_data = sensor_data[-lookback_window:]
            
            # Predict RUL using prediction service
            result = self.prediction_service.predict_rul(
                rig_id=rig_id,
                sensor_data=recent_data,
                model_type=model_type,
                lookback_window=lookback_window
            )
            
            if not result.get("success"):
                return result
            
            predictions = result.get("predictions", [])
            if not predictions:
                return {
                    "success": False,
                    "message": "No predictions returned",
                    "predicted_rul": None
                }
            
            prediction = predictions[0]
            predicted_rul = prediction.get("predicted_rul", 0)
            confidence = prediction.get("confidence", 0)
            
            # Determine alert severity
            severity = self._get_rul_severity(predicted_rul)
            
            # Generate recommendation
            recommendation = self._generate_rul_recommendation(predicted_rul, component)
            
            # Update cache
            if rig_id not in self.rul_cache:
                self.rul_cache[rig_id] = {}
            
            self.rul_cache[rig_id][component] = {
                "predicted_rul": predicted_rul,
                "confidence": confidence,
                "severity": severity,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type
            }
            
            # Save to database
            try:
                self.data_service.save_rul_prediction({
                    "rig_id": rig_id,
                    "component": component,
                    "predicted_rul": predicted_rul,
                    "confidence": confidence,
                    "model_used": model_type,
                    "recommendation": recommendation
                })
            except Exception as e:
                logger.warning(f"Failed to save RUL prediction to database: {e}")
            
            # Check if alert should be sent
            if severity in ["critical", "high"]:
                self._send_rul_alert(rig_id, component, predicted_rul, severity, recommendation)
            
            return {
                "success": True,
                "rig_id": rig_id,
                "component": component,
                "predicted_rul": predicted_rul,
                "confidence": confidence,
                "severity": severity,
                "recommendation": recommendation,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error predicting RUL: {e}")
            return {
                "success": False,
                "message": str(e),
                "predicted_rul": None
            }
    
    def _get_rul_severity(self, predicted_rul: float) -> str:
        """Get severity level based on RUL."""
        if predicted_rul < self.rul_alert_thresholds["critical"]:
            return "critical"
        elif predicted_rul < self.rul_alert_thresholds["high"]:
            return "high"
        elif predicted_rul < self.rul_alert_thresholds["medium"]:
            return "medium"
        elif predicted_rul < self.rul_alert_thresholds["low"]:
            return "low"
        else:
            return "normal"
    
    def _generate_rul_recommendation(self, predicted_rul: float, component: str) -> str:
        """Generate maintenance recommendation based on RUL."""
        if predicted_rul < self.rul_alert_thresholds["critical"]:
            return f"CRITICAL: {component} requires immediate maintenance. RUL: {predicted_rul:.1f} hours"
        elif predicted_rul < self.rul_alert_thresholds["high"]:
            return f"URGENT: Schedule maintenance for {component} within 24 hours. RUL: {predicted_rul:.1f} hours"
        elif predicted_rul < self.rul_alert_thresholds["medium"]:
            return f"Schedule maintenance for {component} within 1 week. RUL: {predicted_rul:.1f} hours"
        elif predicted_rul < self.rul_alert_thresholds["low"]:
            return f"Plan maintenance for {component} within 30 days. RUL: {predicted_rul:.1f} hours"
        else:
            return f"{component} is operating normally. RUL: {predicted_rul:.1f} hours"
    
    def _send_rul_alert(
        self,
        rig_id: str,
        component: str,
        predicted_rul: float,
        severity: str,
        recommendation: str
    ) -> None:
        """Send alert for low RUL."""
        try:
            alert_data = {
                "alert_type": "predictive_maintenance",
                "severity": severity,
                "title": f"Low RUL Alert: {component}",
                "message": recommendation,
                "rig_id": rig_id,
                "sensor_data_snapshot": {
                    "component": component,
                    "predicted_rul": predicted_rul,
                    "severity": severity
                },
                "metadata": {
                    "component": component,
                    "predicted_rul": predicted_rul
                }
            }
            
            # Send via alert management service
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(alert_management_service.create_alert(**alert_data))
                else:
                    loop.run_until_complete(alert_management_service.create_alert(**alert_data))
            except Exception as e:
                logger.warning(f"Failed to send RUL alert via alert service: {e}")
            
            # Broadcast via WebSocket
            try:
                websocket_manager.broadcast_to_rig(
                    rig_id=rig_id,
                    message={
                        "type": "rul_alert",
                        "rig_id": rig_id,
                        "component": component,
                        "predicted_rul": predicted_rul,
                        "severity": severity,
                        "recommendation": recommendation,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast RUL alert via WebSocket: {e}")
        
        except Exception as e:
            logger.error(f"Error sending RUL alert: {e}")
    
    def detect_anomalies_isolation_forest(
        self,
        rig_id: str,
        component: str,
        sensor_data: Optional[Dict[str, Any]] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            rig_id: Rig identifier
            component: Component name
            sensor_data: Current sensor data (if None, fetches latest)
            contamination: Expected proportion of anomalies (0.0 to 0.5)
        
        Returns:
            Anomaly detection result
        """
        if not SKLEARN_AVAILABLE:
            return {
                "success": False,
                "message": "scikit-learn not available. Isolation Forest cannot be used.",
                "is_anomaly": False
            }
        
        try:
            # Get current sensor data if not provided
            if sensor_data is None:
                latest = self.data_service.get_latest_sensor_data(rig_id=rig_id, limit=1)
                if not latest or len(latest) == 0:
                    return {
                        "success": False,
                        "message": "No sensor data available",
                        "is_anomaly": False
                    }
                sensor_data = latest[0]
            
            # Get historical data for training/updating model
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            history = self.data_service.get_historical_data(
                rig_id=rig_id,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not history or len(history) < 50:
                return {
                    "success": False,
                    "message": "Insufficient historical data for anomaly detection",
                    "is_anomaly": False
                }
            
            # Extract features for anomaly detection
            features = self._extract_anomaly_features(history + [sensor_data])
            
            if len(features) < 50:
                return {
                    "success": False,
                    "message": "Insufficient features extracted",
                    "is_anomaly": False
                }
            
            # Get or create Isolation Forest model
            model_key = f"{rig_id}_{component}"
            if model_key not in self.isolation_forest_models:
                # Train new model
                self.isolation_forest_models[model_key] = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                # Use historical data for training
                historical_features = self._extract_anomaly_features(history)
                if len(historical_features) >= 50:
                    self.isolation_forest_models[model_key].fit(historical_features)
            
            model = self.isolation_forest_models[model_key]
            
            # Predict anomaly for current data point
            current_features = self._extract_anomaly_features([sensor_data])
            if len(current_features) == 0:
                return {
                    "success": False,
                    "message": "Failed to extract features from current data",
                    "is_anomaly": False
                }
            
            prediction = model.predict(current_features[0].reshape(1, -1))
            anomaly_score = model.score_samples(current_features[0].reshape(1, -1))[0]
            
            is_anomaly = prediction[0] == -1  # -1 means anomaly, 1 means normal
            
            # Update model periodically with new data
            if len(history) % 100 == 0:  # Retrain every 100 new data points
                historical_features = self._extract_anomaly_features(history)
                if len(historical_features) >= 50:
                    self.isolation_forest_models[model_key] = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    self.isolation_forest_models[model_key].fit(historical_features)
            
            # Store anomaly in history
            if model_key not in self.anomaly_history:
                self.anomaly_history[model_key] = deque(maxlen=1000)
            
            anomaly_entry = {
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "timestamp": datetime.now().isoformat(),
                "sensor_data": sensor_data
            }
            self.anomaly_history[model_key].append(anomaly_entry)
            
            # Save to database
            if is_anomaly:
                try:
                    self.data_service.save_anomaly_detection({
                        "rig_id": rig_id,
                        "component": component,
                        "is_anomaly": True,
                        "anomaly_score": float(anomaly_score),
                        "sensor_data": sensor_data,
                        "detection_method": "isolation_forest"
                    })
                except Exception as e:
                    logger.warning(f"Failed to save anomaly detection to database: {e}")
                
                # Send alert
                self._send_anomaly_alert(rig_id, component, anomaly_score, sensor_data)
            
            return {
                "success": True,
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "component": component,
                "rig_id": rig_id,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                "success": False,
                "message": str(e),
                "is_anomaly": False
            }
    
    def _extract_anomaly_features(self, sensor_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        feature_keys = [
            "wob", "rpm", "torque", "rop", "mud_flow", "mud_pressure",
            "mud_temperature", "vibration", "power_consumption", "hook_load"
        ]
        
        for data_point in sensor_data:
            point_features = []
            for key in feature_keys:
                # Try multiple key variations
                value = (
                    data_point.get(key) or
                    data_point.get(key.upper()) or
                    data_point.get(key.replace("_", " ").title()) or
                    0.0
                )
                try:
                    point_features.append(float(value))
                except (TypeError, ValueError):
                    point_features.append(0.0)
            features.append(point_features)
        
        return np.array(features)
    
    def _send_anomaly_alert(
        self,
        rig_id: str,
        component: str,
        anomaly_score: float,
        sensor_data: Dict[str, Any]
    ) -> None:
        """Send alert for detected anomaly."""
        try:
            # Determine severity based on anomaly score
            # Lower scores indicate more severe anomalies
            if anomaly_score < -0.5:
                severity = "critical"
            elif anomaly_score < -0.3:
                severity = "high"
            elif anomaly_score < -0.1:
                severity = "medium"
            else:
                severity = "low"
            
            alert_data = {
                "alert_type": "anomaly_detection",
                "severity": severity,
                "title": f"Anomaly Detected: {component}",
                "message": f"Anomaly detected in {component} (score: {anomaly_score:.3f}). Review sensor data immediately.",
                "rig_id": rig_id,
                "sensor_data_snapshot": sensor_data,
                "metadata": {
                    "component": component,
                    "anomaly_score": anomaly_score,
                    "detection_method": "isolation_forest"
                }
            }
            
            # Send via alert management service
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(alert_management_service.create_alert(**alert_data))
                else:
                    loop.run_until_complete(alert_management_service.create_alert(**alert_data))
            except Exception as e:
                logger.warning(f"Failed to send anomaly alert via alert service: {e}")
            
            # Broadcast via WebSocket
            try:
                websocket_manager.broadcast_to_rig(
                    rig_id=rig_id,
                    message={
                        "type": "anomaly_alert",
                        "rig_id": rig_id,
                        "component": component,
                        "anomaly_score": anomaly_score,
                        "severity": severity,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast anomaly alert via WebSocket: {e}")
        
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")
    
    def suggest_maintenance_schedule(
        self,
        rig_id: str,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest maintenance schedule based on RUL predictions and anomalies.
        
        Args:
            rig_id: Rig identifier
            component: Optional component name (if None, suggests for all components)
        
        Returns:
            Maintenance schedule suggestions
        """
        try:
            suggestions = []
            
            # Get components to check
            components_to_check = [component] if component else self.components
            
            for comp in components_to_check:
                # Get latest RUL prediction
                rul_result = self.predict_rul_realtime(rig_id, comp)
                
                if not rul_result.get("success"):
                    continue
                
                predicted_rul = rul_result.get("predicted_rul", 0)
                severity = rul_result.get("severity", "normal")
                
                # Get latest anomaly detection
                anomaly_result = self.detect_anomalies_isolation_forest(rig_id, comp)
                has_anomaly = anomaly_result.get("is_anomaly", False)
                
                # Calculate suggested maintenance date
                if predicted_rul > 0:
                    # Suggest maintenance 10% before predicted failure
                    suggested_date = datetime.now() + timedelta(hours=predicted_rul * 0.9)
                else:
                    # If RUL is 0 or negative, suggest immediate maintenance
                    suggested_date = datetime.now()
                
                # Determine priority
                if severity == "critical" or has_anomaly:
                    priority = "urgent"
                elif severity == "high":
                    priority = "high"
                elif severity == "medium":
                    priority = "medium"
                else:
                    priority = "low"
                
                suggestion = {
                    "rig_id": rig_id,
                    "component": comp,
                    "suggested_date": suggested_date.isoformat(),
                    "predicted_rul": predicted_rul,
                    "severity": severity,
                    "has_anomaly": has_anomaly,
                    "priority": priority,
                    "recommendation": rul_result.get("recommendation", ""),
                    "estimated_duration_hours": self._estimate_maintenance_duration(comp),
                    "maintenance_type": self._get_maintenance_type(comp, severity, has_anomaly)
                }
                
                suggestions.append(suggestion)
            
            # Sort by priority and date
            priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
            suggestions.sort(key=lambda x: (priority_order.get(x["priority"], 3), x["suggested_date"]))
            
            return {
                "success": True,
                "rig_id": rig_id,
                "suggestions": suggestions,
                "count": len(suggestions),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error suggesting maintenance schedule: {e}")
            return {
                "success": False,
                "message": str(e),
                "suggestions": []
            }
    
    def _estimate_maintenance_duration(self, component: str) -> float:
        """Estimate maintenance duration in hours."""
        durations = {
            "drill_bit": 4.0,
            "mud_pump": 8.0,
            "top_drive": 12.0,
            "drawworks": 16.0,
            "rotary_table": 6.0,
            "bop": 24.0,
            "mud_system": 8.0,
            "power_system": 12.0
        }
        return durations.get(component, 8.0)
    
    def _get_maintenance_type(self, component: str, severity: str, has_anomaly: bool) -> str:
        """Determine maintenance type."""
        if severity == "critical" or has_anomaly:
            return "emergency"
        elif severity == "high":
            return "preventive"
        else:
            return "scheduled"
    
    def start_monitoring(self) -> None:
        """Start real-time predictive maintenance monitoring."""
        if self.running:
            logger.warning("Predictive maintenance monitoring is already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PredictiveMaintenance-Monitoring"
        )
        self.monitoring_thread.start()
        logger.info("Predictive maintenance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time predictive maintenance monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Predictive maintenance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Get active rigs
                rigs = self._get_active_rigs()
                
                for rig_id in rigs:
                    try:
                        # Predict RUL for all components
                        for component in self.components:
                            try:
                                self.predict_rul_realtime(rig_id, component)
                            except Exception as e:
                                logger.error(f"Error predicting RUL for {rig_id}/{component}: {e}")
                            
                            # Detect anomalies
                            try:
                                self.detect_anomalies_isolation_forest(rig_id, component)
                            except Exception as e:
                                logger.error(f"Error detecting anomalies for {rig_id}/{component}: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring rig {rig_id}: {e}")
                
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Error in predictive maintenance monitoring loop: {e}")
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
predictive_maintenance_service = PredictiveMaintenanceService()

