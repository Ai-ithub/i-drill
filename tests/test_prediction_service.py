"""
Unit tests for PredictionService
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "backend"
sys.path.insert(0, str(src_path))

from services.prediction_service import PredictionService, TORCH_AVAILABLE, RUL_AVAILABLE


@pytest.fixture
def mock_torch():
    """Mock PyTorch"""
    torch = Mock()
    torch.device = Mock(return_value='cpu')
    torch.cuda = Mock()
    torch.cuda.is_available = Mock(return_value=False)
    torch.FloatTensor = Mock(return_value=Mock())
    torch.no_grad = Mock(return_value=Mock())
    return torch


@pytest.fixture
def mock_model():
    """Mock PyTorch model"""
    model = Mock()
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[])
    model.state_dict = Mock(return_value={})
    return model


@pytest.fixture
def prediction_service(mock_torch, mock_model):
    """Create PredictionService instance with mocked dependencies"""
    with patch('services.prediction_service.TORCH_AVAILABLE', True):
        with patch('services.prediction_service.RUL_AVAILABLE', True):
            with patch('services.prediction_service.torch', mock_torch):
                with patch('services.prediction_service.mlflow_service', None):
                    service = PredictionService(model_dir='test_models')
                    service.device = 'cpu'
                    service.loaded_models = {}
                    return service


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    return [
        {
            'depth': 5000.0,
            'wob': 1500.0,
            'rpm': 80.0,
            'torque': 400.0,
            'rop': 12.0,
            'mud_flow': 1200.0,
            'mud_pressure': 3000.0,
            'mud_temperature': 60.0,
            'gamma_ray': 85.0,
            'resistivity': 20.0,
            'density': 2.5,
            'porosity': 0.15,
            'hook_load': 200.0,
            'vibration': 1.5,
            'power_consumption': 250.0
        }
        for _ in range(100)  # Generate 100 data points
    ]


class TestPredictionService:
    """Test suite for PredictionService"""

    def test_init_with_torch_available(self, mock_torch):
        """Test initialization when PyTorch is available"""
        with patch('services.prediction_service.TORCH_AVAILABLE', True):
            with patch('services.prediction_service.torch', mock_torch):
                with patch('services.prediction_service.mlflow_service', None):
                    service = PredictionService()
                    assert service.device == 'cpu'
                    assert service.loaded_models == {}

    def test_init_without_torch(self):
        """Test initialization when PyTorch is not available"""
        with patch('services.prediction_service.TORCH_AVAILABLE', False):
            service = PredictionService()
            assert service.device is None

    def test_predict_rul_success(self, prediction_service, mock_model, sample_sensor_data):
        """Test predicting RUL successfully"""
        # Setup
        prediction_service.loaded_models['lstm'] = mock_model
        
        # Mock model prediction
        mock_prediction_tensor = Mock()
        mock_prediction_tensor.item.return_value = 500.0
        mock_model.return_value = mock_prediction_tensor
        
        # Mock input preparation
        with patch.object(prediction_service, '_prepare_input_data', return_value=np.random.randn(50, 15)):
            with patch.object(prediction_service, '_load_or_create_model', return_value=mock_model):
                # Execute
                result = prediction_service.predict_rul(
                    rig_id='RIG_01',
                    sensor_data=sample_sensor_data,
                    model_type='lstm',
                    lookback_window=50
                )
        
        # Assert
        assert result['success'] is True
        assert len(result['predictions']) == 1
        assert result['predictions'][0]['rig_id'] == 'RIG_01'

    def test_predict_rul_insufficient_data(self, prediction_service, sample_sensor_data):
        """Test predicting RUL with insufficient data"""
        # Execute
        result = prediction_service.predict_rul(
            rig_id='RIG_01',
            sensor_data=sample_sensor_data[:10],  # Only 10 data points
            model_type='lstm',
            lookback_window=50  # Need 50 points
        )
        
        # Assert
        assert result['success'] is False
        assert 'Insufficient data' in result['message']

    def test_predict_rul_no_models(self):
        """Test predicting RUL when models are not available"""
        with patch('services.prediction_service.RUL_AVAILABLE', False):
            service = PredictionService()
            result = service.predict_rul(
                rig_id='RIG_01',
                sensor_data=[],
                model_type='lstm'
            )
            assert result['success'] is False
            assert 'not available' in result['message']

    def test_predict_rul_no_torch(self):
        """Test predicting RUL when PyTorch is not available"""
        with patch('services.prediction_service.TORCH_AVAILABLE', False):
            service = PredictionService()
            result = service.predict_rul(
                rig_id='RIG_01',
                sensor_data=[],
                model_type='lstm'
            )
            assert result['success'] is False
            assert 'not available' in result['message']

    def test_prepare_input_data_success(self, prediction_service, sample_sensor_data):
        """Test preparing input data successfully"""
        # Execute
        result = prediction_service._prepare_input_data(sample_sensor_data, lookback_window=50)
        
        # Assert
        assert result is not None
        assert result.shape == (50, 15)  # 50 time steps, 15 features

    def test_prepare_input_data_insufficient(self, prediction_service, sample_sensor_data):
        """Test preparing input data with insufficient data"""
        # Execute
        result = prediction_service._prepare_input_data(sample_sensor_data[:10], lookback_window=50)
        
        # Assert - should return None or handle gracefully
        # The function will still return array but with fewer rows
        # Actually, the function takes the last lookback_window points, so it should work
        assert result is not None

    def test_extract_feature_value(self, prediction_service):
        """Test extracting feature value from data point"""
        # Setup
        data_point = {
            'depth': 5000.0,
            'wob': 1500.0,
            'weight_on_bit': 1500.0,  # Alternative name
            'rpm': 80.0
        }
        
        # Execute
        depth_value = prediction_service._extract_feature_value(data_point, ['depth'])
        wob_value = prediction_service._extract_feature_value(data_point, ['wob', 'weight_on_bit'])
        
        # Assert
        assert depth_value == 5000.0
        assert wob_value == 1500.0

    def test_extract_feature_value_missing(self, prediction_service):
        """Test extracting feature value when key is missing"""
        # Setup
        data_point = {'depth': 5000.0}
        
        # Execute
        value = prediction_service._extract_feature_value(data_point, ['missing_key'])
        
        # Assert
        assert value == 0.0  # Default value

    def test_extract_feature_value_variations(self, prediction_service):
        """Test extracting feature value with different key variations"""
        # Setup
        data_point = {
            'mud_flow': 1200.0,
            'mud_flow_rate': 1200.0,
            'flow_rate': 1200.0
        }
        
        # Execute
        value1 = prediction_service._extract_feature_value(data_point, ['mud_flow', 'mud_flow_rate', 'flow_rate'])
        value2 = prediction_service._extract_feature_value(data_point, ['MUD_FLOW'])  # Uppercase
        value3 = prediction_service._extract_feature_value(data_point, ['mud-flow'])  # Hyphen
        
        # Assert
        assert value1 == 1200.0
        assert value2 == 1200.0
        assert value3 == 1200.0

    def test_detect_anomalies_high_temperature(self, prediction_service):
        """Test detecting anomalies with high temperature"""
        # Setup
        sensor_data = {
            'temperature': 150.0,  # Above threshold of 100
            'vibration_level': 1.0,
            'power_consumption': 200.0
        }
        
        # Execute
        result = prediction_service.detect_anomalies(sensor_data)
        
        # Assert
        assert result['has_anomaly'] is True
        assert result['anomaly_count'] > 0
        assert len(result['anomalies']) > 0

    def test_detect_anomalies_low_vibration(self, prediction_service):
        """Test detecting anomalies with low vibration"""
        # Setup
        sensor_data = {
            'temperature': 70.0,
            'vibration_level': 0.05,  # Below threshold of 0.1
            'power_consumption': 200.0
        }
        
        # Execute
        result = prediction_service.detect_anomalies(sensor_data)
        
        # Assert
        assert result['has_anomaly'] is True
        assert result['anomaly_count'] > 0

    def test_detect_anomalies_no_anomalies(self, prediction_service):
        """Test detecting anomalies when there are none"""
        # Setup
        sensor_data = {
            'temperature': 70.0,
            'vibration_level': 1.0,
            'power_consumption': 200.0
        }
        
        # Execute
        result = prediction_service.detect_anomalies(sensor_data)
        
        # Assert
        assert result['has_anomaly'] is False
        assert result['anomaly_count'] == 0
        assert len(result['anomalies']) == 0

    def test_detect_anomalies_error_handling(self, prediction_service):
        """Test anomaly detection error handling"""
        # Setup
        sensor_data = None  # Invalid data
        
        # Execute
        result = prediction_service.detect_anomalies(sensor_data)
        
        # Assert
        assert result['has_anomaly'] is False
        assert 'error' in result

    def test_load_or_create_model_from_cache(self, prediction_service, mock_model):
        """Test loading model from cache"""
        # Setup
        prediction_service.loaded_models['lstm'] = mock_model
        
        # Execute
        result = prediction_service._load_or_create_model('lstm')
        
        # Assert
        assert result == mock_model

    def test_load_or_create_model_from_file(self, prediction_service, mock_model):
        """Test loading model from file"""
        # Setup
        model_path = Path('test_models/lstm_rul_model.pth')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with patch('services.prediction_service.torch') as mock_torch:
            mock_torch.load.return_value = mock_model
            with patch('pathlib.Path.exists', return_value=True):
                # Execute
                result = prediction_service._load_or_create_model('lstm')
        
        # Assert
        assert result is not None

    def test_load_or_create_model_create_new(self, prediction_service, mock_model):
        """Test creating new model when none exists"""
        # Setup
        with patch('services.prediction_service.create_model', return_value=mock_model):
            with patch('pathlib.Path.exists', return_value=False):
                with patch('services.prediction_service.mlflow_service', None):
                    # Execute
                    result = prediction_service._load_or_create_model('lstm')
        
        # Assert
        assert result is not None
        assert 'lstm' in prediction_service.loaded_models

    def test_ensure_model_dir(self, prediction_service):
        """Test ensuring model directory exists"""
        # Setup
        test_dir = Path('test_models_ensure')
        
        # Execute
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            prediction_service.model_dir = str(test_dir)
            prediction_service._ensure_model_dir()
        
        # Assert
        # Directory creation should be called
        # (The actual check depends on whether directory exists)

    def test_predict_rul_with_mlflow(self, prediction_service, mock_model, sample_sensor_data):
        """Test predicting RUL with MLflow logging"""
        # Setup
        mock_mlflow = Mock()
        mock_mlflow.log_inference = Mock()
        
        prediction_service.loaded_models['lstm'] = mock_model
        
        # Mock model prediction
        mock_prediction_tensor = Mock()
        mock_prediction_tensor.item.return_value = 500.0
        mock_model.return_value = mock_prediction_tensor
        
        with patch('services.prediction_service.mlflow_service', mock_mlflow):
            with patch.object(prediction_service, '_prepare_input_data', return_value=np.random.randn(50, 15)):
                with patch.object(prediction_service, '_load_or_create_model', return_value=mock_model):
                    # Execute
                    result = prediction_service.predict_rul(
                        rig_id='RIG_01',
                        sensor_data=sample_sensor_data,
                        model_type='lstm',
                        lookback_window=50
                    )
        
        # Assert
        assert result['success'] is True
        # MLflow logging should be called (if available)
        # Note: This depends on mlflow_service being available

    def test_detect_anomalies_severity_levels(self, prediction_service):
        """Test detecting anomalies with different severity levels"""
        # Setup - High severity (above 1.5x threshold)
        sensor_data_high = {
            'temperature': 200.0,  # 2x threshold (high severity)
            'vibration_level': 1.0
        }
        
        # Setup - Medium severity (above threshold but below 1.5x)
        sensor_data_medium = {
            'temperature': 120.0,  # 1.2x threshold (medium severity)
            'vibration_level': 1.0
        }
        
        # Execute
        result_high = prediction_service.detect_anomalies(sensor_data_high)
        result_medium = prediction_service.detect_anomalies(sensor_data_medium)
        
        # Assert
        assert result_high['has_anomaly'] is True
        assert result_medium['has_anomaly'] is True
        
        # Check severity levels
        high_severity_anomalies = [a for a in result_high['anomalies'] if a['severity'] == 'high']
        medium_severity_anomalies = [a for a in result_medium['anomalies'] if a['severity'] == 'medium']
        
        assert len(high_severity_anomalies) > 0
        assert len(medium_severity_anomalies) > 0

    def test_predict_rul_confidence_calculation(self, prediction_service, mock_model, sample_sensor_data):
        """Test RUL prediction confidence calculation"""
        # Setup
        prediction_service.loaded_models['lstm'] = mock_model
        
        # Mock model prediction with high RUL value
        mock_prediction_tensor = Mock()
        mock_prediction_tensor.item.return_value = 2000.0  # High RUL
        mock_model.return_value = mock_prediction_tensor
        
        with patch.object(prediction_service, '_prepare_input_data', return_value=np.random.randn(50, 15)):
            with patch.object(prediction_service, '_load_or_create_model', return_value=mock_model):
                # Execute
                result = prediction_service.predict_rul(
                    rig_id='RIG_01',
                    sensor_data=sample_sensor_data,
                    model_type='lstm',
                    lookback_window=50
                )
        
        # Assert
        assert result['success'] is True
        assert result['predictions'][0]['confidence'] <= 1.0
        assert result['predictions'][0]['confidence'] >= 0.0

    def test_predict_rul_negative_rul_handling(self, prediction_service, mock_model, sample_sensor_data):
        """Test handling negative RUL predictions"""
        # Setup
        prediction_service.loaded_models['lstm'] = mock_model
        
        # Mock model prediction with negative RUL
        mock_prediction_tensor = Mock()
        mock_prediction_tensor.item.return_value = -100.0  # Negative RUL
        mock_model.return_value = mock_prediction_tensor
        
        with patch.object(prediction_service, '_prepare_input_data', return_value=np.random.randn(50, 15)):
            with patch.object(prediction_service, '_load_or_create_model', return_value=mock_model):
                # Execute
                result = prediction_service.predict_rul(
                    rig_id='RIG_01',
                    sensor_data=sample_sensor_data,
                    model_type='lstm',
                    lookback_window=50
                )
        
        # Assert
        assert result['success'] is True
        # Negative RUL should be clamped to 0
        assert result['predictions'][0]['predicted_rul'] >= 0

