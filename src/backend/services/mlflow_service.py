"""
MLflow Service
Manages ML model versioning, tracking, and registry
"""
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore
    import mlflow.pytorch  # type: ignore
    import mlflow.sklearn  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow is not installed. Model lifecycle features are disabled.")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "i-drill-models")


class MLflowService:
    """Service for MLflow operations"""

    def __init__(self):
        """Initialize MLflow service"""
        self.client = None
        self.experiment = None
        self.experiment_id = "0"

        if not MLFLOW_AVAILABLE:
            return

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            try:
                self.experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
                if self.experiment is None:
                    self.experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                    self.experiment = mlflow.get_experiment(self.experiment_id)
                else:
                    self.experiment_id = self.experiment.experiment_id
            except Exception as experiment_error:
                logger.warning(f"Could not create/get experiment: {experiment_error}")
                self.experiment_id = "0"

            self.client = MlflowClient()

            logger.info(f"MLflow service initialized: {MLFLOW_TRACKING_URI}")
            logger.info(f"Experiment: {MLFLOW_EXPERIMENT_NAME} (ID: {self.experiment_id})")
        except Exception as init_error:
            logger.error(f"Error initializing MLflow service: {init_error}")
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
        Log a model to MLflow tracking and registry.
        
        Logs model artifacts, metrics, parameters, and tags to MLflow.
        Registers the model in the MLflow model registry.
        
        Args:
            model: Model object to log (PyTorch, scikit-learn, or ONNX)
            model_name: Name for model registration
            framework: Framework type ("pytorch", "sklearn", "onnx")
            metrics: Optional dictionary of metrics to log
            params: Optional dictionary of parameters to log
            tags: Optional dictionary of tags to set
            
        Returns:
            MLflow run ID if successful, None otherwise
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; skipping model logging")
            return None

        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                if params:
                    mlflow.log_params(params)
                if metrics:
                    mlflow.log_metrics(metrics)
                if tags:
                    mlflow.set_tags(tags)

                if framework == "pytorch":
                    mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=model_name)
                elif framework == "sklearn":
                    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
                elif framework == "onnx":
                    mlflow.onnx.log_model(model, artifact_path="model", registered_model_name=model_name)
                else:
                    logger.error(f"Unsupported framework: {framework}")
                    return None

                run_id = run.info.run_id
                logger.info(f"Model logged: {model_name} (Run ID: {run_id})")
                return run_id
        except Exception as log_error:
            logger.error(f"Error logging model: {log_error}")
            return None

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = "Production"
    ):
        """
        Load a model from MLflow model registry.
        
        Loads a model using MLflow's pyfunc interface, which supports
        multiple frameworks.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (optional)
            stage: Model stage to load from (default: "Production")
                   Ignored if version is specified
            
        Returns:
            Loaded model object, or None if loading fails
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot load model")
            return None

        try:
            model_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded: {model_name} from {model_uri}")
            return model
        except Exception as load_error:
            logger.error(f"Error loading model: {load_error}")
            return None

    def load_pytorch_model(
        self,
        model_name: str,
        stage: str = "Production"
    ):
        """
        Load a PyTorch model from the MLflow registry.
        
        Loads a PyTorch model specifically, preserving PyTorch-specific
        functionality and methods.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to load from (default: "Production")
            
        Returns:
            Loaded PyTorch model, or None if loading fails
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot load PyTorch model")
            return None

        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded PyTorch model from MLflow: {model_uri}")
            return model
        except Exception as load_error:
            logger.warning(f"Unable to load PyTorch model '{model_name}' from MLflow: {load_error}")
            return None

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            List of dictionaries containing version information:
            - version: Version number
            - stage: Current stage
            - run_id: MLflow run ID
            - status: Version status
            - creation_timestamp: When version was created
            - last_updated_timestamp: When version was last updated
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot query model versions")
            return []

        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                }
                for version in versions
            ]
        except Exception as versions_error:
            logger.error(f"Error getting model versions: {versions_error}")
            return []

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """
        Transition a model version to a different stage.
        
        Moves a model version between stages (e.g., Staging -> Production).
        
        Args:
            model_name: Name of the registered model
            version: Version number to transition
            stage: Target stage (e.g., "Staging", "Production", "Archived")
            
        Returns:
            True if transition succeeded, False otherwise
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot transition model stage")
            return False

        try:
            self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
            return True
        except Exception as transition_error:
            logger.error(f"Error transitioning model stage: {transition_error}")
            return False

    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get all registered models in the MLflow registry.
        
        Returns:
            List of dictionaries containing model information:
            - name: Model name
            - creation_timestamp: When model was created
            - last_updated_timestamp: When model was last updated
            - description: Model description
            - latest_versions: List of latest versions per stage
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot list registered models")
            return []

        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": [
                        {"version": version.version, "stage": version.current_stage}
                        for version in model.latest_versions
                    ],
                }
                for model in models
            ]
        except Exception as models_error:
            logger.error(f"Error getting registered models: {models_error}")
            return []

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the active MLflow run.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for time-series metrics
        """
        if not MLFLOW_AVAILABLE:
            logger.debug("MLflow unavailable; skipping metric logging")
            return

        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
        except Exception as metrics_error:
            logger.error(f"Error logging metrics: {metrics_error}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file) to MLflow.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the artifact directory
        """
        if not MLFLOW_AVAILABLE:
            logger.debug("MLflow unavailable; skipping artifact logging")
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Artifact logged: {local_path}")
        except Exception as artifact_error:
            logger.error(f"Error logging artifact: {artifact_error}")

    def log_inference(
        self,
        model_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an inference event to MLflow.
        
        Creates a nested run for tracking inference metrics and parameters.
        
        Args:
            model_name: Name of the model used for inference
            metrics: Dictionary of inference metrics (e.g., latency, accuracy)
            params: Optional dictionary of inference parameters
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; skipping inference logging")
            return

        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"inference::{model_name}",
                nested=True,
            ):
                if params:
                    mlflow.log_params(params)
                mlflow.log_metrics(metrics)
        except Exception as inference_error:
            logger.debug(f"Failed to log inference to MLflow: {inference_error}")

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a registered model from MLflow registry.
        
        Permanently deletes the model and all its versions.
        
        Args:
            model_name: Name of the registered model to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        if not MLFLOW_AVAILABLE or self.client is None:
            logger.debug("MLflow unavailable; cannot delete model")
            return False

        try:
            self.client.delete_registered_model(model_name)
            logger.info(f"Model deleted: {model_name}")
            return True
        except Exception as delete_error:
            logger.error(f"Error deleting model: {delete_error}")
            return False


# Global MLflow service instance
try:
    mlflow_service = MLflowService() if MLFLOW_AVAILABLE else None
except Exception as service_error:
    logger.warning(f"MLflow service initialization failed: {service_error}")
    mlflow_service = None
