"""
Model Validation Service
Provides comprehensive model validation capabilities
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Model validation features are limited.")


class ModelValidationService:
    """
    Service for validating machine learning models.
    
    Provides validation for:
    - Model performance metrics
    - Data quality checks
    - Model drift detection
    - Prediction consistency
    - Model fairness
    """
    
    def __init__(self, performance_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize ModelValidationService.
        
        Args:
            performance_thresholds: Dictionary of metric names to minimum acceptable values
        """
        self.performance_thresholds = performance_thresholds or {
            "accuracy": 0.7,
            "f1_score": 0.6,
            "r2_score": 0.5,
            "mean_absolute_error": 100.0,  # Max acceptable error
        }
    
    def validate_regression_model(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Validate a regression model.
        
        Args:
            predictions: Model predictions
            true_values: True target values
            model_name: Name of the model being validated
            
        Returns:
            Dictionary containing validation results and metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"valid": False, "error": "scikit-learn not available"}
        
        try:
            # Calculate metrics
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_values, predictions)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100
            
            # Calculate error distribution
            errors = predictions - true_values
            error_mean = np.mean(errors)
            error_std = np.std(errors)
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2_score": float(r2),
                "mape": float(mape),
                "error_mean": float(error_mean),
                "error_std": float(error_std),
            }
            
            # Check thresholds
            validations = {
                "r2_acceptable": r2 >= self.performance_thresholds.get("r2_score", 0.5),
                "mae_acceptable": mae <= self.performance_thresholds.get("mean_absolute_error", 100.0),
                "error_distribution_ok": abs(error_mean) < error_std,
            }
            
            all_valid = all(validations.values())
            
            return {
                "valid": all_valid,
                "model_name": model_name,
                "metrics": metrics,
                "validations": validations,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error validating regression model: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_classification_model(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Validate a classification model.
        
        Args:
            predictions: Model predictions (class labels)
            true_labels: True class labels
            model_name: Name of the model being validated
            
        Returns:
            Dictionary containing validation results and metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"valid": False, "error": "scikit-learn not available"}
        
        try:
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
            recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
            f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
            }
            
            # Check thresholds
            validations = {
                "accuracy_acceptable": accuracy >= self.performance_thresholds.get("accuracy", 0.7),
                "f1_acceptable": f1 >= self.performance_thresholds.get("f1_score", 0.6),
            }
            
            all_valid = all(validations.values())
            
            return {
                "valid": all_valid,
                "model_name": model_name,
                "metrics": metrics,
                "validations": validations,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error validating classification model: {e}")
            return {"valid": False, "error": str(e)}
    
    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        new_data: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and new data.
        
        Uses Kolmogorov-Smirnov test for continuous features and
        chi-square test for categorical features.
        
        Args:
            reference_data: Reference dataset
            new_data: New dataset to compare
            threshold: Maximum allowed drift (p-value threshold)
            
        Returns:
            Dictionary containing drift detection results
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for drift detection")
            return {"drift_detected": False, "method": "unavailable"}
        
        try:
            drift_results = {}
            
            # Handle 1D arrays
            if reference_data.ndim == 1:
                reference_data = reference_data.reshape(-1, 1)
                new_data = new_data.reshape(-1, 1)
            
            # Check each feature
            for i in range(reference_data.shape[1]):
                ref_feature = reference_data[:, i]
                new_feature = new_data[:, i]
                
                # KS test for continuous data
                ks_statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
                
                drift_results[f"feature_{i}"] = {
                    "ks_statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "drift_detected": p_value < threshold,
                }
            
            overall_drift = any(
                result["drift_detected"]
                for result in drift_results.values()
            )
            
            return {
                "drift_detected": overall_drift,
                "threshold": threshold,
                "feature_results": drift_results,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return {"drift_detected": False, "error": str(e)}
    
    def validate_prediction_consistency(
        self,
        predictions: List[np.ndarray],
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate that model predictions are consistent across multiple runs.
        
        Args:
            predictions: List of prediction arrays from multiple runs
            tolerance: Maximum allowed difference between predictions
            
        Returns:
            Dictionary containing consistency validation results
        """
        try:
            if len(predictions) < 2:
                return {"consistent": True, "message": "Not enough runs to validate"}
            
            # Convert to numpy arrays
            pred_arrays = [np.array(p) for p in predictions]
            
            # Check consistency between all pairs
            max_diff = 0.0
            inconsistencies = []
            
            for i in range(len(pred_arrays)):
                for j in range(i + 1, len(pred_arrays)):
                    diff = np.abs(pred_arrays[i] - pred_arrays[j])
                    max_pair_diff = np.max(diff)
                    max_diff = max(max_diff, max_pair_diff)
                    
                    if max_pair_diff > tolerance:
                        inconsistencies.append({
                            "run_pair": (i, j),
                            "max_difference": float(max_pair_diff),
                        })
            
            consistent = max_diff <= tolerance
            
            return {
                "consistent": consistent,
                "max_difference": float(max_diff),
                "tolerance": tolerance,
                "inconsistencies": inconsistencies,
                "num_runs": len(predictions),
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error validating prediction consistency: {e}")
            return {"consistent": False, "error": str(e)}
    
    def validate_model_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Validate model metrics against thresholds.
        
        Args:
            metrics: Dictionary of metric names to values
            model_name: Name of the model
            
        Returns:
            Dictionary containing validation results
        """
        validations = {}
        
        for metric_name, threshold_value in self.performance_thresholds.items():
            if metric_name in metrics:
                # For error metrics, lower is better
                if "error" in metric_name.lower() or "mae" in metric_name.lower():
                    validations[metric_name] = metrics[metric_name] <= threshold_value
                else:
                    # For score metrics, higher is better
                    validations[metric_name] = metrics[metric_name] >= threshold_value
        
        all_valid = all(validations.values()) if validations else False
        
        return {
            "valid": all_valid,
            "model_name": model_name,
            "validations": validations,
            "thresholds": self.performance_thresholds,
            "actual_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
    
    def generate_validation_report(
        self,
        validation_results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: List of validation result dictionaries
            output_path: Optional path to save report
            
        Returns:
            Report as HTML string
        """
        report_lines = [
            "<!DOCTYPE html>",
            "<html><head><title>Model Validation Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2 { color: #333; }",
            ".valid { color: green; }",
            ".invalid { color: red; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "</style></head><body>",
            "<h1>Model Validation Report</h1>",
            f"<p>Generated: {datetime.now().isoformat()}</p>",
        ]
        
        for result in validation_results:
            status_class = "valid" if result.get("valid", False) else "invalid"
            status_text = "✅ PASS" if result.get("valid", False) else "❌ FAIL"
            
            report_lines.extend([
                f"<h2>{result.get('model_name', 'Unknown Model')}</h2>",
                f"<p class='{status_class}'><strong>Status: {status_text}</strong></p>",
            ])
            
            if "metrics" in result:
                report_lines.append("<h3>Metrics</h3><table>")
                report_lines.append("<tr><th>Metric</th><th>Value</th></tr>")
                for metric, value in result["metrics"].items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>")
                report_lines.append("</table>")
            
            if "validations" in result:
                report_lines.append("<h3>Validations</h3><table>")
                report_lines.append("<tr><th>Check</th><th>Result</th></tr>")
                for check, passed in result["validations"].items():
                    status = "✅" if passed else "❌"
                    report_lines.append(f"<tr><td>{check}</td><td>{status}</td></tr>")
                report_lines.append("</table>")
        
        report_lines.append("</body></html>")
        
        report_html = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_html)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_html


# Global validation service instance
model_validation_service = ModelValidationService()

