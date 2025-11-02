"""
Prediction Service for RUL and maintenance predictions
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Try to import torch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Prediction features will be limited.")

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from rul_prediction.models import create_model
    RUL_AVAILABLE = True
except ImportError:
    RUL_AVAILABLE = False
    logger.warning("RUL prediction models not available")


class PredictionService:
    """Service for making predictions"""
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or "models"
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.loaded_models = {}
        if TORCH_AVAILABLE:
            self._ensure_model_dir()
    
    def _ensure_model_dir(self):
        """Ensure model directory exists"""
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
    
    def predict_rul(
        self,
        rig_id: str,
        sensor_data: List[Dict[str, Any]],
        model_type: str = "lstm",
        lookback_window: int = 50
    ) -> Dict[str, Any]:
        """
        Predict remaining useful life
        
        Args:
            rig_id: Rig ID
            sensor_data: List of sensor data points
            model_type: Type of model to use (lstm, transformer, cnn_lstm)
            lookback_window: Number of historical points to use
            
        Returns:
            Dictionary with prediction results
        """
        if not RUL_AVAILABLE:
            return {
                'success': False,
                'rig_id': rig_id,
                'predicted_rul_hours': 0,
                'confidence_score': 0,
                'prediction_timestamp': datetime.now(),
                'model_used': model_type,
                'message': 'RUL models not available'
            }
        
        if not TORCH_AVAILABLE or not RUL_AVAILABLE:
            return {
                'success': False,
                'rig_id': rig_id,
                'predicted_rul_hours': 0,
                'confidence_score': 0,
                'prediction_timestamp': datetime.now(),
                'model_used': model_type,
                'message': 'PyTorch or RUL models not available'
            }
        
        try:
            # Check if we have enough data
            if len(sensor_data) < lookback_window:
                return {
                    'success': False,
                    'rig_id': rig_id,
                    'predicted_rul_hours': 0,
                    'confidence_score': 0,
                    'prediction_timestamp': datetime.now(),
                    'model_used': model_type,
                    'message': f'Insufficient data: need {lookback_window} points, got {len(sensor_data)}'
                }
            
            # Load or create model
            model = self._load_or_create_model(model_type)
            if model is None:
                return {
                    'success': False,
                    'rig_id': rig_id,
                    'predicted_rul_hours': 0,
                    'confidence_score': 0,
                    'prediction_timestamp': datetime.now(),
                    'model_used': model_type,
                    'message': 'Failed to load model'
                }
            
            # Prepare input data
            input_data = self._prepare_input_data(sensor_data, lookback_window)
            if input_data is None:
                return {
                    'success': False,
                    'rig_id': rig_id,
                    'predicted_rul_hours': 0,
                    'confidence_score': 0,
                    'prediction_timestamp': datetime.now(),
                    'model_used': model_type,
                    'message': 'Failed to prepare input data'
                }
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
                prediction = model(input_tensor)
                predicted_rul = float(prediction.item())
                
                # Calculate confidence (simplified)
                confidence_score = min(1.0, max(0.0, predicted_rul / 1000))  # Normalize
            
            logger.info(f"RUL prediction for {rig_id}: {predicted_rul:.2f} hours")
            
            return {
                'success': True,
                'rig_id': rig_id,
                'predicted_rul_hours': max(0, predicted_rul),
                'confidence_score': confidence_score,
                'prediction_timestamp': datetime.now(),
                'model_used': model_type,
                'message': None
            }
            
        except Exception as e:
            logger.error(f"Error predicting RUL: {e}")
            return {
                'success': False,
                'rig_id': rig_id,
                'predicted_rul_hours': 0,
                'confidence_score': 0,
                'prediction_timestamp': datetime.now(),
                'model_used': model_type,
                'message': f'Prediction error: {str(e)}'
            }
    
    def _load_or_create_model(self, model_type: str):
        """Load or create a model"""
        try:
            # Check if model is already loaded
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
            
            # Try to load from file
            model_path = Path(self.model_dir) / f"{model_type}_rul_model.pth"
            if model_path.exists():
                model = torch.load(model_path, map_location=self.device)
                self.loaded_models[model_type] = model
                return model
            
            # Create a new model (for demo/testing)
            # In production, this should fail if no model exists
            logger.warning(f"No trained model found for {model_type}, creating new model")
            input_dim = 15  # Number of sensor features
            model = create_model(model_type, input_dim)
            model.to(self.device)
            self.loaded_models[model_type] = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            return None
    
    def _prepare_input_data(self, sensor_data: List[Dict[str, Any]], lookback_window: int) -> Optional[np.ndarray]:
        """
        Prepare input data for model
        
        Args:
            sensor_data: List of sensor data points
            lookback_window: Number of points to use
            
        Returns:
            NumPy array of shape (lookback_window, num_features)
        """
        try:
            # Get the last lookback_window points
            recent_data = sensor_data[-lookback_window:]
            
            # Extract numeric features
            feature_names = [
                'depth', 'wob', 'rpm', 'torque', 'rop',
                'mud_flow_rate', 'mud_pressure', 'mud_temperature',
                'mud_density', 'mud_viscosity', 'mud_ph',
                'gamma_ray', 'resistivity', 'power_consumption',
                'vibration_level'
            ]
            
            # Build feature array
            features = []
            for data_point in recent_data:
                point_features = []
                for name in feature_names:
                    # Convert to snake_case if needed
                    snake_name = name.lower()
                    camel_name = name[0].upper() + name[1:]
                    
                    value = data_point.get(name) or data_point.get(snake_name) or data_point.get(camel_name)
                    if value is None:
                        value = 0.0
                    point_features.append(float(value))
                
                features.append(point_features)
            
            if len(features) != lookback_window:
                logger.error(f"Expected {lookback_window} features, got {len(features)}")
                return None
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing input data: {e}")
            return None
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data
        
        Args:
            sensor_data: Single sensor data point
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            anomalies = []
            
            # Simple threshold-based anomaly detection
            thresholds = {
                'temperature': {'high': 100, 'low': 40},
                'motor_temperature': {'high': 95, 'low': 60},
                'vibration_level': {'high': 2.0, 'low': 0.1},
                'power_consumption': {'high': 300, 'low': 100}
            }
            
            for field, limits in thresholds.items():
                value = sensor_data.get(field)
                if value is not None:
                    if value > limits['high']:
                        anomalies.append({
                            'field': field,
                            'value': value,
                            'threshold': limits['high'],
                            'severity': 'high' if value > limits['high'] * 1.5 else 'medium',
                            'message': f'{field} exceeds maximum threshold'
                        })
                    elif value < limits['low']:
                        anomalies.append({
                            'field': field,
                            'value': value,
                            'threshold': limits['low'],
                            'severity': 'medium',
                            'message': f'{field} below minimum threshold'
                        })
            
            return {
                'has_anomaly': len(anomalies) > 0,
                'anomaly_count': len(anomalies),
                'anomalies': anomalies
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                'has_anomaly': False,
                'anomaly_count': 0,
                'anomalies': [],
                'error': str(e)
            }

