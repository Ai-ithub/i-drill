from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from services.mlflow_service import mlflow_service

logger = logging.getLogger(__name__)


class TrainingPipelineService:
    """Service for orchestrating predictive model training and deployment."""

    def __init__(self) -> None:
        self.mlflow_service = mlflow_service

    def _mlflow_available(self) -> bool:
        return mlflow is not None and self.mlflow_service is not None

    def start_training_job(
        self,
        model_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger a lightweight training run tracked in MLflow."""
        if not self._mlflow_available():
            return {"success": False, "message": "MLflow is not configured on this environment"}

        experiment_name = experiment_name or "i-drill-training"
        params = parameters or {}

        try:
            mlflow.set_experiment(experiment_name)  # type: ignore[call-arg]
            with mlflow.start_run(run_name=f"train::{model_name}") as run:  # type: ignore[call-arg]
                mlflow.log_params(params)  # type: ignore[call-arg]
                metrics = {"placeholder_accuracy": 0.9, "placeholder_loss": 0.1}
                mlflow.log_metrics(metrics)  # type: ignore[call-arg]

                run_id = run.info.run_id
                return {
                    "success": True,
                    "run_id": run_id,
                    "experiment_name": experiment_name,
                    "metrics": metrics,
                }
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Training job failed: {exc}")
            return {"success": False, "message": str(exc)}

    def promote_model(self, model_name: str, version: str, stage: str) -> Dict[str, Any]:
        if not self._mlflow_available():
            return {"success": False, "message": "MLflow is not configured on this environment"}

        try:
            success = self.mlflow_service.transition_model_stage(model_name, version, stage)
            return {"success": success}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to promote model {model_name} v{version}: {exc}")
            return {"success": False, "message": str(exc)}

    def list_registered_models(self) -> List[Dict[str, Any]]:
        if not self._mlflow_available():
            return []
        try:
            return self.mlflow_service.get_registered_models()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to list registered models: {exc}")
            return []

    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        if not self._mlflow_available():
            return []
        try:
            return self.mlflow_service.get_model_versions(model_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to list model versions for {model_name}: {exc}")
            return []


training_pipeline_service = TrainingPipelineService()
