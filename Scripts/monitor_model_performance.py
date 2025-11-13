#!/usr/bin/env python3
"""
Script to monitor model performance in production
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlflow
    from src.backend.services.mlflow_service import mlflow_service
    from src.backend.services.model_validation_service import model_validation_service
except ImportError as e:
    print(f"⚠️ Required packages not available: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_production_metrics(
    model_name: str,
    time_window_hours: int = 24
) -> Dict[str, Any]:
    """Collect production metrics for a model"""
    # This is a placeholder - implement based on your monitoring infrastructure
    logger.info(f"Collecting production metrics for {model_name}")
    
    # In a real implementation, this would query:
    # - Prometheus metrics
    # - Application logs
    # - Database records
    # - Inference API metrics
    
    return {
        "model_name": model_name,
        "time_window_hours": time_window_hours,
        "total_predictions": 0,
        "average_latency_ms": 0.0,
        "error_rate": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "success_rate": 1.0,
        "timestamp": datetime.now().isoformat(),
    }


def check_model_drift(
    model_name: str,
    reference_data_path: str,
    production_data_path: str
) -> Dict[str, Any]:
    """Check for data drift in production"""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.error("numpy/pandas not available")
        return {}
    
    # Load reference data
    ref_data = pd.read_csv(reference_data_path).values
    
    # Load production data
    prod_data = pd.read_csv(production_data_path).values
    
    # Detect drift
    drift_result = model_validation_service.detect_data_drift(ref_data, prod_data)
    
    return drift_result


def check_model_performance(
    model_name: str,
    performance_thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Check if model performance meets thresholds"""
    # Get latest metrics from MLflow
    if not mlflow_service or mlflow_service.client is None:
        logger.error("MLflow service not available")
        return {}
    
    try:
        client = mlflow_service.client
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            logger.warning(f"No versions found for model: {model_name}")
            return {}
        
        # Get latest production version
        prod_version = None
        for version in model_versions:
            if version.current_stage == "Production":
                prod_version = version
                break
        
        if not prod_version:
            logger.warning(f"No Production version found for model: {model_name}")
            return {}
        
        # Get run metrics
        run = client.get_run(prod_version.run_id)
        metrics = run.data.metrics
        
        # Validate against thresholds
        validation = model_validation_service.validate_model_metrics(
            metrics, model_name
        )
        
        return {
            "model_name": model_name,
            "version": prod_version.version,
            "validation": validation,
        }
    
    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return {}


def monitor_model(
    model_name: str,
    monitoring_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Main monitoring function"""
    logger.info(f"🔍 Monitoring model: {model_name}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }
    
    # Collect production metrics
    time_window = monitoring_config.get("time_window_hours", 24)
    prod_metrics = collect_production_metrics(model_name, time_window)
    results["checks"]["production_metrics"] = prod_metrics
    
    # Check performance thresholds
    thresholds = monitoring_config.get("performance_thresholds", {})
    if thresholds:
        perf_check = check_model_performance(model_name, thresholds)
        results["checks"]["performance"] = perf_check
    
    # Check for data drift (if configured)
    drift_config = monitoring_config.get("drift_detection", {})
    if drift_config.get("enabled", False):
        drift_result = check_model_drift(
            model_name,
            drift_config.get("reference_data"),
            drift_config.get("production_data"),
        )
        results["checks"]["drift"] = drift_result
    
    # Determine overall health
    health_checks = []
    if "performance" in results["checks"]:
        validation = results["checks"]["performance"].get("validation", {})
        health_checks.append(validation.get("valid", False))
    
    if "drift" in results["checks"]:
        drift_detected = results["checks"]["drift"].get("drift_detected", False)
        health_checks.append(not drift_detected)
    
    results["healthy"] = all(health_checks) if health_checks else True
    
    return results


def log_monitoring_to_mlflow(
    model_name: str,
    monitoring_results: Dict[str, Any],
    experiment_name: str = "i-drill-monitoring"
):
    """Log monitoring results to MLflow"""
    if not mlflow_service or mlflow_service.client is None:
        logger.warning("MLflow service not available")
        return
    
    try:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"monitor-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log metrics
            if "production_metrics" in monitoring_results.get("checks", {}):
                metrics = monitoring_results["checks"]["production_metrics"]
                mlflow.log_metrics({
                    "total_predictions": metrics.get("total_predictions", 0),
                    "average_latency_ms": metrics.get("average_latency_ms", 0.0),
                    "error_rate": metrics.get("error_rate", 0.0),
                    "success_rate": metrics.get("success_rate", 1.0),
                })
            
            # Log tags
            mlflow.set_tags({
                "model_name": model_name,
                "monitoring_type": "production",
                "healthy": str(monitoring_results.get("healthy", True)),
            })
            
            logger.info("✅ Monitoring results logged to MLflow")
    
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")


def main():
    parser = argparse.ArgumentParser(description="Monitor model performance")
    parser.add_argument("--model_name", required=True, help="Model name to monitor")
    parser.add_argument("--config", type=str, help="Monitoring configuration JSON file")
    parser.add_argument("--experiment", default="i-drill-monitoring", help="MLflow experiment name")
    parser.add_argument("--interval", type=int, default=3600, help="Monitoring interval in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            try:
                results = monitor_model(args.model_name, config)
                log_monitoring_to_mlflow(args.model_name, results, args.experiment)
                
                if not results.get("healthy", True):
                    logger.warning(f"⚠️ Model {args.model_name} is unhealthy!")
                
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(args.interval)
    else:
        results = monitor_model(args.model_name, config)
        log_monitoring_to_mlflow(args.model_name, results, args.experiment)
        
        # Print results
        print(json.dumps(results, indent=2))
        
        # Exit with error code if unhealthy
        if not results.get("healthy", True):
            sys.exit(1)


if __name__ == "__main__":
    main()

