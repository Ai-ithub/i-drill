"""
Tests for Training Pipeline Service
Tests model training pipeline operations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the service
import sys
from pathlib import Path
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


@pytest.mark.unit
@pytest.mark.service
class TestTrainingPipelineService:
    """Tests for TrainingPipelineService"""
    
    @pytest.fixture
    def service(self):
        """Create TrainingPipelineService instance"""
        with patch('services.training_pipeline_service.mlflow') as mock_mlflow:
            with patch('services.training_pipeline_service.mlflow_service') as mock_mlflow_service:
                from services.training_pipeline_service import TrainingPipelineService
                service = TrainingPipelineService()
                service.mlflow_service = mock_mlflow_service
                return service
    
    def test_service_initialization(self, service):
        """Test service initializes correctly"""
        assert service is not None
        assert hasattr(service, 'mlflow_service')
    
    @patch('services.training_pipeline_service.mlflow')
    def test_start_training_job_success(self, mock_mlflow, service):
        """Test starting a training job successfully"""
        mock_mlflow.set_experiment = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        
        result = service.start_training_job(
            model_name="test_model",
            parameters={"learning_rate": 0.001},
            experiment_name="test-experiment"
        )
        
        assert result["success"] is True
        assert "run_id" in result
        assert result["run_id"] == "test-run-id"
    
    @patch('services.training_pipeline_service.mlflow', None)
    def test_start_training_job_mlflow_unavailable(self, service):
        """Test starting training job when MLflow unavailable"""
        result = service.start_training_job(
            model_name="test_model",
            parameters={}
        )
        
        assert result["success"] is False
        assert "not configured" in result["message"].lower()
    
    def test_promote_model_success(self, service):
        """Test promoting a model"""
        service.mlflow_service.transition_model_stage = MagicMock(return_value=True)
        
        result = service.promote_model(
            model_name="test_model",
            version="1",
            stage="Production"
        )
        
        assert result["success"] is True
        service.mlflow_service.transition_model_stage.assert_called_once_with(
            "test_model", "1", "Production"
        )
    
    def test_promote_model_failure(self, service):
        """Test promoting a model with failure"""
        service.mlflow_service.transition_model_stage = MagicMock(
            side_effect=Exception("Promotion failed")
        )
        
        result = service.promote_model(
            model_name="test_model",
            version="1",
            stage="Production"
        )
        
        assert result["success"] is False
        assert "message" in result
    
    def test_list_registered_models(self, service):
        """Test listing registered models"""
        mock_models = [
            {"name": "model1", "version": "1"},
            {"name": "model2", "version": "1"}
        ]
        service.mlflow_service.get_registered_models = MagicMock(
            return_value=mock_models
        )
        
        result = service.list_registered_models()
        
        assert len(result) == 2
        assert result == mock_models
    
    def test_list_model_versions(self, service):
        """Test listing model versions"""
        mock_versions = [
            {"version": "1", "stage": "Production"},
            {"version": "2", "stage": "Staging"}
        ]
        service.mlflow_service.get_model_versions = MagicMock(
            return_value=mock_versions
        )
        
        result = service.list_model_versions("test_model")
        
        assert len(result) == 2
        assert result == mock_versions
    
    @patch('services.training_pipeline_service.mlflow')
    def test_trigger_training_success(self, mock_mlflow, service):
        """Test triggering training successfully"""
        mock_mlflow.set_experiment = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        
        result = service.trigger_training(
            model_name="test_model",
            experiment_name="test-experiment",
            params={"param1": "value1"}
        )
        
        assert result["success"] is True
        assert "run_id" in result
        assert result["run_id"] == "test-run-id"
    
    @patch('services.training_pipeline_service.mlflow', None)
    def test_trigger_training_mlflow_unavailable(self, service):
        """Test triggering training when MLflow unavailable"""
        result = service.trigger_training(
            model_name="test_model"
        )
        
        assert result["success"] is False
        assert "not configured" in result["message"].lower()


@pytest.mark.integration
class TestTrainingPipelineServiceIntegration:
    """Integration tests for TrainingPipelineService"""
    
    def test_training_pipeline_service_singleton(self):
        """Test that training_pipeline_service is a singleton"""
        from services.training_pipeline_service import training_pipeline_service
        assert training_pipeline_service is not None

