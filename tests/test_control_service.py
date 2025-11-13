"""
Tests for Control Service
Tests integration with drilling control system
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the service
import sys
from pathlib import Path
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from services.control_service import ControlService, control_service


@pytest.mark.unit
@pytest.mark.service
class TestControlService:
    """Tests for ControlService"""
    
    @pytest.fixture
    def service(self):
        """Create ControlService instance"""
        return ControlService()
    
    def test_service_initialization(self, service):
        """Test service initializes correctly"""
        assert service.available is True
    
    def test_is_available(self, service):
        """Test availability check"""
        assert service.is_available() is True
    
    def test_apply_parameter_change_success(self, service):
        """Test successful parameter change application"""
        result = service.apply_parameter_change(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            new_value=120.0,
            metadata={"user": "test_user"}
        )
        
        assert result["success"] is True
        assert "message" in result
        assert "applied_at" in result
        assert result["applied_at"] is not None
    
    def test_apply_parameter_change_validation_failure(self, service):
        """Test parameter change with invalid value"""
        # Test value exceeding maximum
        result = service.apply_parameter_change(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            new_value=1000.0,  # Exceeds max of 500
            metadata={}
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "validation" in result["error"].lower() or "maximum" in result["error"].lower()
    
    def test_apply_parameter_change_below_minimum(self, service):
        """Test parameter change with value below minimum"""
        result = service.apply_parameter_change(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            new_value=-10.0,  # Below minimum of 0
            metadata={}
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_validate_parameter_value_valid(self, service):
        """Test validation of valid parameter value"""
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            value=100.0
        )
        
        assert result["valid"] is True
        assert result["reason"] is None
    
    def test_validate_parameter_value_invalid_max(self, service):
        """Test validation of value exceeding maximum"""
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            value=600.0  # Exceeds max of 500
        )
        
        assert result["valid"] is False
        assert "maximum" in result["reason"].lower()
    
    def test_validate_parameter_value_invalid_min(self, service):
        """Test validation of value below minimum"""
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm",
            value=-5.0  # Below min of 0
        )
        
        assert result["valid"] is False
        assert "minimum" in result["reason"].lower()
    
    def test_validate_parameter_value_non_numeric(self, service):
        """Test validation of non-numeric value"""
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="status",
            value="active"  # Non-numeric value
        )
        
        # Non-numeric values should be valid for some parameters
        assert result["valid"] is True
    
    def test_get_parameter_value(self, service):
        """Test getting parameter value from control system"""
        result = service.get_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="rpm"
        )
        
        # In mock implementation, returns None
        # In production, would query actual control system
        assert result is None or isinstance(result, (int, float, str))
    
    def test_apply_parameter_change_with_exception(self, service):
        """Test parameter change application with exception"""
        with patch.object(service, '_validate_parameter_value', side_effect=Exception("Test error")):
            result = service.apply_parameter_change(
                rig_id="RIG_01",
                component="drilling",
                parameter="rpm",
                new_value=100.0
            )
            
            assert result["success"] is False
            assert "error" in result
    
    def test_validate_parameter_wob(self, service):
        """Test validation of WOB parameter"""
        # Valid WOB
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="wob",
            value=100000.0
        )
        assert result["valid"] is True
        
        # Invalid WOB (too high)
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="wob",
            value=600000.0  # Exceeds max
        )
        assert result["valid"] is False
    
    def test_validate_parameter_torque(self, service):
        """Test validation of torque parameter"""
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="torque",
            value=50000.0
        )
        assert result["valid"] is True
        
        result = service._validate_parameter_value(
            rig_id="RIG_01",
            component="drilling",
            parameter="torque",
            value=150000.0  # Exceeds max
        )
        assert result["valid"] is False


@pytest.mark.integration
class TestControlServiceIntegration:
    """Integration tests for ControlService"""
    
    def test_control_service_singleton(self):
        """Test that control_service is a singleton"""
        from services.control_service import control_service
        assert control_service is not None
        assert isinstance(control_service, ControlService)

