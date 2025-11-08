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


from services.mlflow_service import mlflow_service  # noqa: E402


FEATURE_MAPPINGS: Dict[str, List[str]] = {
    "depth": ["depth"],
    "wob": ["wob", "weight_on_bit"],
    "rpm": ["rpm"],
    "torque": ["torque"],
    "rop": ["rop", "rate_of_penetration"],
    "mud_flow": ["mud_flow", "mud_flow_rate", "flow_rate"],
    "mud_pressure": ["mud_pressure", "standpipe_pressure", "pressure"],
    "mud_temperature": ["mud_temperature", "temperature"],
    "gamma_ray": ["gamma_ray"],
    "resistivity": ["resistivity"],
    "density": ["density"],
    "porosity": ["porosity"],
    "hook_load": ["hook_load"],
    "vibration": ["vibration", "vibration_level"],
    "power_consumption": ["power_consumption"],
}

FEATURE_ORDER: List[str] = list(FEATURE_MAPPINGS.keys())


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
                'predictions': [],
                'message': 'RUL models not available'
            }
        
        if not TORCH_AVAILABLE or not RUL_AVAILABLE:
            return {
                'success': False,
                'predictions': [],
                'message': 'PyTorch or RUL models not available'
            }
        
        try:
            # Check if we have enough data
            if len(sensor_data) < lookback_window:
                return {
                    'success': False,
                    'predictions': [],
                    'message': f'Insufficient data: need {lookback_window} points, got {len(sensor_data)}'
                }
            
            # Load or create model
            model = self._load_or_create_model(model_type)
            if model is None:
                return {
                    'success': False,
                    'predictions': [],
                    'message': 'Failed to load model'
                }
            
            # Prepare input data
            input_data = self._prepare_input_data(sensor_data, lookback_window)
            if input_data is None:
                return {
                    'success': False,
                    'predictions': [],
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
            
            prediction_entry = {
                'rig_id': rig_id,
                'component': 'general',
                'predicted_rul': max(0, predicted_rul),
                'confidence': confidence_score,
                'timestamp': datetime.now(),
                'model_used': model_type,
                'recommendation': None
            }
            
            response_payload = {
                'success': True,
                'predictions': [prediction_entry],
                'message': None
            }
            
            if mlflow_service:
                try:
                    mlflow_service.log_inference(
                        model_name=model_type,
                        metrics={
                            'predicted_rul': max(0, predicted_rul),
                            'confidence': confidence_score
                        },
                        params={
                            'rig_id': rig_id,
                            'lookback_window': lookback_window,
                            'data_points': len(sensor_data)
                        }
                    )
                except Exception as log_error:
                    logger.debug(f"Failed to log inference to MLflow: {log_error}")
            
            return response_payload
            
        except Exception as e:
            logger.error(f"Error predicting RUL: {e}")
            return {
                'success': False,
                'predictions': [],
                'message': f'Prediction error: {str(e)}'
            }
    
    def _load_or_create_model(self, model_type: str):
        """Load or create a model"""
        try:
            # Check if model is already loaded
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
            
            # Try MLflow registry
            if mlflow_service:
                registry_model = mlflow_service.load_pytorch_model(model_type)
                if registry_model is not None:
                    try:
                        if hasattr(registry_model, "to"):
                            registry_model = registry_model.to(self.device)
                        registry_model.eval()
                        self.loaded_models[model_type] = registry_model
                        return registry_model
                    except Exception as e:
                        logger.warning(f"Failed to prepare MLflow model '{model_type}': {e}")
            
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
            # Build feature array
            features = []
            for data_point in recent_data:
                point_features = []
                for feature_key in FEATURE_ORDER:
                    value = self._extract_feature_value(
                        data_point,
                        FEATURE_MAPPINGS[feature_key]
                    )
                    point_features.append(value)
                
                features.append(point_features)
            
            if len(features) != lookback_window:
                logger.error(f"Expected {lookback_window} features, got {len(features)}")
                return None
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing input data: {e}")
            return None
    
    @staticmethod
    def _extract_feature_value(data_point: Dict[str, Any], candidates: List[str]) -> float:
        """Extract a numeric feature value from data point based on candidate keys"""
        candidate_keys = set()
        for raw_key in candidates:
            if not raw_key:
                continue
            snake = raw_key.lower()
            candidate_keys.update({
                raw_key,
                snake,
                snake.replace('-', '_'),
                snake.replace(' ', '_'),
                raw_key.upper(),
                ''.join(part.title() for part in snake.split('_')),
            })
            camel = ''.join(part.title() for part in snake.split('_'))
            if camel:
                candidate_keys.add(camel[0].lower() + camel[1:])
        
        for key in candidate_keys:
            value = data_point.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        
        return 0.0
    
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

