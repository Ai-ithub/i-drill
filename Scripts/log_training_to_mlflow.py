#!/usr/bin/env python3
"""
Script to log training results to MLflow
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlflow
    from src.backend.services.mlflow_service import mlflow_service
except ImportError:
    print("⚠️ MLflow not available. Install with: pip install mlflow")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_metrics(model_path: str) -> Dict[str, Any]:
    """Load training metrics from model directory"""
    metrics_file = Path(model_path) / "training_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    # Try to extract from log files
    log_file = Path(model_path) / "training.log"
    if log_file.exists():
        # Simple parsing - can be improved
        metrics = {}
        with open(log_file, "r") as f:
            for line in f:
                if "accuracy" in line.lower():
                    try:
                        metrics["accuracy"] = float(line.split(":")[-1].strip())
                    except:
                        pass
        return metrics
    
    return {}


def log_to_mlflow(
    model_type: str,
    model_name: str,
    model_path: str,
    experiment_name: str,
    additional_params: Dict[str, Any] = None
) -> str:
    """Log training results to MLflow"""
    
    if not mlflow_service or mlflow_service.client is None:
        logger.error("MLflow service not available")
        sys.exit(1)
    
    # Set experiment
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set experiment: {e}")
        experiment = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Load metrics
    metrics = load_training_metrics(model_path)
    
    # Default parameters
    params = {
        "model_type": model_type,
        "model_name": model_name,
        "framework": "stable_baselines3" if model_type in ["ppo", "sac"] else "pytorch",
    }
    
    if additional_params:
        params.update(additional_params)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"train-{model_type}-{model_name}") as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)
        else:
            logger.warning("No metrics found, logging defaults")
            mlflow.log_metrics({"placeholder_accuracy": 0.9, "placeholder_loss": 0.1})
        
        # Log tags
        mlflow.set_tags({
            "model_type": model_type,
            "model_name": model_name,
            "training_pipeline": "automated",
        })
        
        # Log model artifacts
        if (model_path_obj / "model").exists():
            mlflow.log_artifacts(str(model_path_obj / "model"), artifact_path="model")
        elif model_path_obj.is_file() and model_path_obj.suffix in [".zip", ".pkl", ".pt", ".pth"]:
            mlflow.log_artifact(str(model_path_obj), artifact_path="model")
        
        # Log training logs if available
        for log_file in model_path_obj.glob("*.log"):
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        
        # Log any config files
        for config_file in model_path_obj.glob("*config*.json"):
            mlflow.log_artifact(str(config_file), artifact_path="config")
        
        run_id = run.info.run_id
        logger.info(f"✅ Logged training run to MLflow: {run_id}")
        
        # Register model
        try:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_name
            )
            logger.info(f"✅ Registered model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not register model: {e}")
        
        return run_id


def main():
    parser = argparse.ArgumentParser(description="Log training results to MLflow")
    parser.add_argument("--model_type", required=True, help="Model type (ppo, sac, lstm, etc.)")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--model_path", default="./models", help="Path to model directory")
    parser.add_argument("--experiment", default="i-drill-training", help="MLflow experiment name")
    parser.add_argument("--params", type=str, help="Additional parameters as JSON string")
    
    args = parser.parse_args()
    
    additional_params = {}
    if args.params:
        try:
            additional_params = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in --params")
            sys.exit(1)
    
    run_id = log_to_mlflow(
        model_type=args.model_type,
        model_name=args.model_name,
        model_path=args.model_path,
        experiment_name=args.experiment,
        additional_params=additional_params
    )
    
    print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()

