"""
Tests for Model Deployment Service
Tests model deployment operations
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


@pytest.mark.unit
@pytest.mark.service
class TestModelDeploymentService:
    """Tests for ModelDeploymentService"""
    
    @pytest.fixture
    def service(self):
        """Create ModelDeploymentService instance"""
        with patch('services.model_deployment_service.MLFLOW_AVAILABLE', True):
            with patch('services.model_deployment_service.MlflowClient') as mock_client:
                with patch('services.model_deployment_service.mlflow') as mock_mlflow:
                    mock_mlflow.set_tracking_uri = MagicMock()
                    from services.model_deployment_service import ModelDeploymentService
                    service = ModelDeploymentService()
                    service.mlflow_client = mock_client.return_value
                    return service
    
    def test_service_initialization(self, service):
        """Test service initializes correctly"""
        assert service is not None
        assert hasattr(service, 'mlflow_client')
    
    def test_deploy_model_canary(self, service):
        """Test canary deployment"""
        # Mock MLflow client
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Staging"
        mock_version.creation_timestamp = datetime.now().timestamp() * 1000
        
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[mock_version]
        )
        
        result = service.deploy_model(
            model_name="test_model",
            stage="Staging",
            strategy="canary",
            environment="staging"
        )
        
        assert result["success"] is True
        assert result["strategy"] == "canary"
        assert "deployment_id" in result
    
    def test_deploy_model_blue_green(self, service):
        """Test blue-green deployment"""
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Staging"
        
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[mock_version]
        )
        
        result = service.deploy_model(
            model_name="test_model",
            stage="Staging",
            strategy="blue_green",
            environment="staging"
        )
        
        assert result["success"] is True
        assert result["strategy"] == "blue_green"
    
    def test_deploy_model_rolling(self, service):
        """Test rolling deployment"""
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Staging"
        
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[mock_version]
        )
        
        result = service.deploy_model(
            model_name="test_model",
            stage="Staging",
            strategy="rolling",
            environment="staging"
        )
        
        assert result["success"] is True
        assert result["strategy"] == "rolling"
    
    def test_deploy_model_not_found(self, service):
        """Test deployment when model not found"""
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[]
        )
        
        result = service.deploy_model(
            model_name="nonexistent_model",
            stage="Staging",
            strategy="canary"
        )
        
        assert result["success"] is False
        assert "not found" in result["message"].lower()
    
    def test_deploy_model_unknown_strategy(self, service):
        """Test deployment with unknown strategy"""
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Staging"
        
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[mock_version]
        )
        
        result = service.deploy_model(
            model_name="test_model",
            stage="Staging",
            strategy="unknown_strategy"
        )
        
        assert result["success"] is False
        assert "unknown" in result["message"].lower()
    
    def test_rollback_deployment(self, service):
        """Test rollback deployment"""
        # Mock current production version
        current_version = MagicMock()
        current_version.version = "2"
        current_version.current_stage = "Production"
        
        # Mock previous version
        previous_version = MagicMock()
        previous_version.version = "1"
        previous_version.current_stage = "Archived"
        
        service.mlflow_client.search_model_versions = MagicMock(
            side_effect=[
                [current_version],  # Production versions
                [current_version, previous_version]  # All versions
            ]
        )
        
        result = service.rollback_deployment("test_model")
        
        assert result["success"] is True
        assert "from_version" in result
        assert "to_version" in result
    
    def test_rollback_no_production_version(self, service):
        """Test rollback when no production version exists"""
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[]
        )
        
        result = service.rollback_deployment("test_model")
        
        assert result["success"] is False
        assert "no production" in result["message"].lower()
    
    def test_get_deployment_status(self, service):
        """Test getting deployment status"""
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.creation_timestamp = datetime.now().timestamp() * 1000
        
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[mock_version]
        )
        
        result = service.get_deployment_status("test_model", "production")
        
        assert result["success"] is True
        assert result["deployed"] is True
        assert result["version"] == "1"
    
    def test_get_deployment_status_not_deployed(self, service):
        """Test getting status when model not deployed"""
        service.mlflow_client.search_model_versions = MagicMock(
            return_value=[]
        )
        
        result = service.get_deployment_status("test_model", "production")
        
        assert result["success"] is True
        assert result["deployed"] is False


@pytest.mark.integration
class TestModelDeploymentServiceIntegration:
    """Integration tests for ModelDeploymentService"""
    
    def test_deployment_service_singleton(self):
        """Test that model_deployment_service is a singleton"""
        with patch('services.model_deployment_service.MLFLOW_AVAILABLE', False):
            from services.model_deployment_service import model_deployment_service
            assert model_deployment_service is not None

