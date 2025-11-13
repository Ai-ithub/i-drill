#!/usr/bin/env python3
"""
Script to check model quality before deployment
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    from src.backend.services.model_validation_service import model_validation_service
except ImportError as e:
    print(f"⚠️ Required packages not available: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_thresholds(config_path: str) -> dict:
    """Load quality thresholds from config file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config.get("validation", {}).get("performance_thresholds", {})


def check_model_quality(
    metrics: dict,
    thresholds: dict
) -> tuple[bool, list[str]]:
    """Check if model meets quality thresholds"""
    issues = []
    
    for metric_name, threshold_value in thresholds.items():
        if metric_name not in metrics:
            issues.append(f"Missing metric: {metric_name}")
            continue
        
        metric_value = metrics[metric_name]
        
        # For error metrics, lower is better
        if "error" in metric_name.lower() or "mae" in metric_name.lower():
            if metric_value > threshold_value:
                issues.append(
                    f"{metric_name}: {metric_value:.4f} exceeds threshold "
                    f"{threshold_value:.4f} (higher is worse)"
                )
        else:
            # For score metrics, higher is better
            if metric_value < threshold_value:
                issues.append(
                    f"{metric_name}: {metric_value:.4f} below threshold "
                    f"{threshold_value:.4f} (higher is better)"
                )
    
    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Check model quality")
    parser.add_argument("--metrics_file", required=True, help="JSON file with model metrics")
    parser.add_argument("--thresholds_file", default="./config/mlops_config.yaml", help="Config file with thresholds")
    
    args = parser.parse_args()
    
    # Load metrics
    with open(args.metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Load thresholds
    thresholds = load_thresholds(args.thresholds_file)
    
    # Check quality
    is_acceptable, issues = check_model_quality(metrics, thresholds)
    
    if is_acceptable:
        logger.info("✅ Model quality checks passed")
        sys.exit(0)
    else:
        logger.error("❌ Model quality checks failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()

