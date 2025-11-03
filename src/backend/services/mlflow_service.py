"""
MLflow Service
Manages ML model versioning, tracking, and registry
"""
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, Any, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "i-drill-models")


class MLflowService:
    """Service for MLflow operations"""
    
    def __init__(self):
        """Initialize MLflow service"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Create or get experiment
            try:
                self.experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
                if self.experiment is None:
                    self.experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                    self.experiment = mlflow.get_experiment(self.experiment_id)
                else:
                    self.experiment_id = self.experiment.experiment_id
            except Exception as e:
                logger.warning(f"Could not create/get experiment: {e}")
                self.experiment_id = "0"  # Default experiment
            
            # Initialize MLflow client
            self.client = MlflowClient()
            
            logger.info(f"MLflow service initialized: {MLFLOW_TRACKING_URI}")
            logger.info(f"Experiment: {MLFLOW_EXPERIMENT_NAME} (ID: {self.experiment_id})")
            
        except Exception as e:
            logger.error(f"Error initializing MLflow service: {e}")
            self.client = None
    
    def log_model(
        self,
        model,
        model_name: str,
        framework: str = "pytorch",
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Log a model to MLflow
        
        Args:
            model: Model object (PyTorch, sklearn, etc.)
            model_name: Name of the model
            framework: Framework type ("pytorch", "sklearn", "onnx")
            metrics: Model metrics to log
            params: Model parameters to log
            tags: Tags for the model
            
        Returns:
            Run ID if successful, None otherwise
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return None
        
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Log model based on framework
                if framework == "pytorch":
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                elif framework == "sklearn":
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                elif framework == "onnx":
                    mlflow.onnx.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                else:
                    logger.error(f"Unsupported framework: {framework}")
                    return None
                
                run_id = run.info.run_id
                logger.info(f"Model logged: {model_name} (Run ID: {run_id})")
                
                return run_id
                
        except Exception as e:
            logger.error(f"Error logging model: {e}")
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = "Production"
    ):
        """
        Load a model from MLflow registry
        
        Args:
            model_name: Name of the registered model
            version: Specific version number (optional)
            stage: Model stage ("Production", "Staging", "Archived")
            
        Returns:
            Loaded model object
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return None
        
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Model loaded: {model_name} from {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_model_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of model version information
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return []
        
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_list = []
            for version in versions:
                version_list.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp
                })
            
            return version_list
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Transition a model version to a different stage
        
        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage ("Production", "Staging", "Archived")
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False
    
    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get all registered models
        
        Returns:
            List of registered model information
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return []
        
        try:
            models = self.client.search_registered_models()
            
            model_list = []
            for model in models:
                model_list.append({
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage
                        }
                        for v in model.latest_versions
                    ]
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error getting registered models: {e}")
            return []
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to current run
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking over time
        """
        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
                
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ):
        """
        Log an artifact (file) to MLflow
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifacts directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Artifact logged: {local_path}")
            
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a registered model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("MLflow client not initialized")
            return False
        
        try:
            self.client.delete_registered_model(model_name)
            logger.info(f"Model deleted: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False


# Global MLflow service instance
try:
    mlflow_service = MLflowService()
except Exception as e:
    logger.warning(f"MLflow service initialization failed: {e}")
    mlflow_service = None

