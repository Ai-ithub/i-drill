"""
Unit tests for Integration Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.integration_service import IntegrationService


class TestIntegrationService:
    """Tests for IntegrationService"""
    
    @pytest.fixture
    def integration_service(self):
        """Create IntegrationService instance"""
        return IntegrationService()
    
    @pytest.fixture
    def sample_sensor_data(self):
        """Sample sensor data for testing"""
        return {
            "rig_id": "RIG_01",
            "timestamp": "2024-01-01T00:00:00",
            "depth": 5000.0,
            "wob": 15000.0,
            "rpm": 120.0,
            "torque": 8000.0,
            "rop": 50.0,
            "mud_flow": 500.0,
            "mud_pressure": 3000.0
        }
    
    def test_init(self, integration_service):
        """Test IntegrationService initialization"""
        assert integration_service.rl_service is not None
        assert integration_service.dvr_service is not None
        assert integration_service.data_service is not None
    
    def test_process_sensor_data_for_rl_success(self, integration_service, sample_sensor_data):
        """Test processing sensor data for RL successfully"""
        # Mock DVR service to return success
        integration_service.dvr_service.process_record = Mock(return_value={
            "success": True,
            "data": sample_sensor_data,
            "message": "Validated"
        })
        
        # Mock RL service
        integration_service.rl_service.get_state = Mock(return_value={
            "state": [1.0, 2.0, 3.0],
            "info": {}
        })
        
        result = integration_service.process_sensor_data_for_rl(
            sample_sensor_data,
            apply_to_rl=True
        )
        
        assert result["success"] is True
        assert "dvr_result" in result
        assert "rl_state" in result
    
    def test_process_sensor_data_for_rl_dvr_failure(self, integration_service, sample_sensor_data):
        """Test processing sensor data when DVR validation fails"""
        # Mock DVR service to return failure
        integration_service.dvr_service.process_record = Mock(return_value={
            "success": False,
            "message": "Validation failed"
        })
        
        result = integration_service.process_sensor_data_for_rl(
            sample_sensor_data,
            apply_to_rl=True
        )
        
        assert result["success"] is False
        assert "DVR validation failed" in result["message"]
        assert result["rl_state"] is None
    
    def test_process_sensor_data_for_rl_without_apply(self, integration_service, sample_sensor_data):
        """Test processing sensor data without applying to RL"""
        # Mock DVR service
        integration_service.dvr_service.process_record = Mock(return_value={
            "success": True,
            "data": sample_sensor_data
        })
        
        result = integration_service.process_sensor_data_for_rl(
            sample_sensor_data,
            apply_to_rl=False
        )
        
        assert result["success"] is True
        assert result["rl_state"] is None
    
    def test_validate_rl_action(self, integration_service):
        """Test validating RL action through DVR"""
        action = {
            "rig_id": "RIG_01",
            "parameter": "rpm",
            "value": 150.0
        }
        
        # Mock DVR service
        integration_service.dvr_service.validate_parameter_change = Mock(return_value={
            "valid": True,
            "message": "Action validated"
        })
        
        result = integration_service.validate_rl_action(action)
        
        assert result["valid"] is True
        assert "message" in result
    
    def test_get_integrated_state(self, integration_service):
        """Test getting integrated state from RL and DVR"""
        rig_id = "RIG_01"
        
        # Mock services
        integration_service.rl_service.get_state = Mock(return_value={
            "state": [1.0, 2.0, 3.0]
        })
        integration_service.dvr_service.get_latest_stats = Mock(return_value={
            "stats": {"valid": 100, "invalid": 5}
        })
        
        result = integration_service.get_integrated_state(rig_id)
        
        assert "rl_state" in result
        assert "dvr_stats" in result
    
    def test_apply_rl_action_with_validation(self, integration_service):
        """Test applying RL action with DVR validation"""
        action = {
            "rig_id": "RIG_01",
            "parameter": "rpm",
            "value": 150.0
        }
        
        # Mock services
        integration_service.dvr_service.validate_parameter_change = Mock(return_value={
            "valid": True
        })
        integration_service.rl_service.apply_action = Mock(return_value={
            "success": True,
            "new_state": [1.0, 2.0, 3.0]
        })
        
        result = integration_service.apply_rl_action_with_validation(action)
        
        assert result["success"] is True
        assert "action_result" in result

