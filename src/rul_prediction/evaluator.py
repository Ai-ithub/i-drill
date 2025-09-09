#!/usr/bin/env python3
"""
Evaluation Module for RUL Prediction Models

This module provides comprehensive evaluation metrics and visualization
tools for remaining useful life prediction models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from scipy import stats

class RULEvaluator:
    """
    Comprehensive evaluator for RUL prediction models
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto',
                 save_dir: str = './evaluation_results'):
        """
        Initialize RUL Evaluator
        
        Args:
            model: Trained PyTorch model
            device: Device to use for evaluation
            save_dir: Directory to save evaluation results
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Results directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            Tuple of (predictions, true_values)
        """
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model(data).squeeze()
                
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(targets.numpy())
                
        return np.array(predictions), np.array(true_values)
    
    def calculate_metrics(self, predictions: np.ndarray, 
                         true_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            predictions: Predicted RUL values
            true_values: True RUL values
            
        Returns:
            Dictionary containing various metrics
        """
        # Basic regression metrics
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100
        
        # RUL-specific metrics
        rul_score = self._calculate_rul_score(predictions, true_values)
        early_predictions = np.sum(predictions < true_values)
        late_predictions = np.sum(predictions > true_values)
        
        # Accuracy within thresholds
        accuracy_10 = np.mean(np.abs(predictions - true_values) <= 10) * 100
        accuracy_20 = np.mean(np.abs(predictions - true_values) <= 20) * 100
        accuracy_30 = np.mean(np.abs(predictions - true_values) <= 30) * 100
        
        # Statistical measures
        correlation = np.corrcoef(predictions, true_values)[0, 1]
        
        # Bias and variance
        bias = np.mean(predictions - true_values)
        variance = np.var(predictions - true_values)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'rul_score': rul_score,
            'early_predictions': early_predictions,
            'late_predictions': late_predictions,
            'accuracy_10': accuracy_10,
            'accuracy_20': accuracy_20,
            'accuracy_30': accuracy_30,
            'correlation': correlation,
            'bias': bias,
            'variance': variance,
            'total_samples': len(predictions)
        }
    
    def _calculate_rul_score(self, predictions: np.ndarray, 
                           true_values: np.ndarray) -> float:
        """
        Calculate RUL-specific scoring function (NASA's PHM08 challenge metric)
        
        Args:
            predictions: Predicted RUL values
            true_values: True RUL values
            
        Returns:
            RUL score (lower is better)
        """
        errors = predictions - true_values
        score = 0
        
        for error in errors:
            if error < 0:  # Early prediction (conservative)
                score += np.exp(-error / 13) - 1
            else:  # Late prediction (risky)
                score += np.exp(error / 10) - 1
                
        return score / len(errors)
    
    def evaluate(self, test_loader: DataLoader, 
                detailed_analysis: bool = True) -> Dict[str, any]:
        """
        Comprehensive evaluation of the model
        
        Args:
            test_loader: Test data loader
            detailed_analysis: Whether to perform detailed analysis
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Starting model evaluation...")
        
        # Generate predictions
        predictions, true_values = self.predict(test_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, true_values)
        
        # Log main metrics
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        self.logger.info(f"  MAE: {metrics['mae']:.4f}")
        self.logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        self.logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        self.logger.info(f"  RUL Score: {metrics['rul_score']:.4f}")
        self.logger.info(f"  Accuracy (±10): {metrics['accuracy_10']:.2f}%")
        
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'true_values': true_values
        }
        
        if detailed_analysis:
            # Detailed analysis
            results.update(self._detailed_analysis(predictions, true_values))
            
            # Generate visualizations
            self._create_visualizations(predictions, true_values, metrics)
            
        # Save results
        self._save_results(results)
        
        return results
    
    def _detailed_analysis(self, predictions: np.ndarray, 
                          true_values: np.ndarray) -> Dict[str, any]:
        """
        Perform detailed analysis of predictions
        
        Args:
            predictions: Predicted values
            true_values: True values
            
        Returns:
            Dictionary with detailed analysis results
        """
        errors = predictions - true_values
        abs_errors = np.abs(errors)
        
        # Error distribution analysis
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'q25_error': np.percentile(errors, 25),
            'q75_error': np.percentile(errors, 75),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors)
        }
        
        # RUL range analysis
        rul_ranges = {
            'low_rul': (true_values <= 50),
            'medium_rul': (true_values > 50) & (true_values <= 200),
            'high_rul': (true_values > 200)
        }
        
        range_analysis = {}
        for range_name, mask in rul_ranges.items():
            if np.sum(mask) > 0:
                range_predictions = predictions[mask]
                range_true = true_values[mask]
                range_analysis[range_name] = {
                    'count': np.sum(mask),
                    'rmse': np.sqrt(mean_squared_error(range_true, range_predictions)),
                    'mae': mean_absolute_error(range_true, range_predictions),
                    'mape': np.mean(np.abs((range_true - range_predictions) / (range_true + 1e-8))) * 100
                }
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(predictions, true_values)
        
        return {
            'error_statistics': error_stats,
            'range_analysis': range_analysis,
            'confidence_intervals': confidence_intervals
        }
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray,
                                      true_values: np.ndarray,
                                      confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for predictions
        
        Args:
            predictions: Predicted values
            true_values: True values
            confidence: Confidence level
            
        Returns:
            Dictionary with confidence interval bounds
        """
        errors = predictions - true_values
        alpha = 1 - confidence
        
        # Assuming normal distribution of errors
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Calculate bounds
        lower_bound = mean_error - stats.norm.ppf(1 - alpha/2) * std_error
        upper_bound = mean_error + stats.norm.ppf(1 - alpha/2) * std_error
        
        return {
            'confidence_level': confidence,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'mean_error': mean_error,
            'std_error': std_error
        }
    
    def _create_visualizations(self, predictions: np.ndarray, 
                             true_values: np.ndarray, 
                             metrics: Dict[str, float]):
        """
        Create comprehensive visualizations
        
        Args:
            predictions: Predicted values
            true_values: True values
            metrics: Calculated metrics
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Predictions vs True Values
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(true_values, predictions, alpha=0.6, s=20)
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'Predictions vs True Values\nR² = {metrics["r2_score"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = plt.subplot(3, 3, 2)
        residuals = predictions - true_values
        plt.scatter(true_values, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True RUL')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}')
        plt.grid(True, alpha=0.3)
        
        # 4. Absolute error vs True RUL
        ax4 = plt.subplot(3, 3, 4)
        abs_errors = np.abs(residuals)
        plt.scatter(true_values, abs_errors, alpha=0.6, s=20)
        plt.xlabel('True RUL')
        plt.ylabel('Absolute Error')
        plt.title(f'Absolute Error vs True RUL\nMAE: {metrics["mae"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 5. Q-Q plot for normality check
        ax5 = plt.subplot(3, 3, 5)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality Check)')
        plt.grid(True, alpha=0.3)
        
        # 6. Cumulative error distribution
        ax6 = plt.subplot(3, 3, 6)
        sorted_abs_errors = np.sort(abs_errors)
        cumulative_prob = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        plt.plot(sorted_abs_errors, cumulative_prob * 100)
        plt.xlabel('Absolute Error')
        plt.ylabel('Cumulative Probability (%)')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 7. Error by RUL ranges
        ax7 = plt.subplot(3, 3, 7)
        rul_bins = [0, 50, 100, 200, 500, np.inf]
        bin_labels = ['0-50', '50-100', '100-200', '200-500', '500+']
        bin_errors = []
        
        for i in range(len(rul_bins) - 1):
            mask = (true_values >= rul_bins[i]) & (true_values < rul_bins[i + 1])
            if np.sum(mask) > 0:
                bin_errors.append(np.mean(abs_errors[mask]))
            else:
                bin_errors.append(0)
        
        plt.bar(bin_labels, bin_errors, alpha=0.7)
        plt.xlabel('RUL Range')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error by RUL Range')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Time series of predictions (if applicable)
        ax8 = plt.subplot(3, 3, 8)
        sample_indices = np.random.choice(len(predictions), min(1000, len(predictions)), replace=False)
        sample_indices = np.sort(sample_indices)
        plt.plot(sample_indices, true_values[sample_indices], 'b-', label='True RUL', alpha=0.7)
        plt.plot(sample_indices, predictions[sample_indices], 'r-', label='Predicted RUL', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('RUL')
        plt.title('Sample Predictions Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Metrics summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        metrics_text = f"""
        Evaluation Metrics:
        
        RMSE: {metrics['rmse']:.4f}
        MAE: {metrics['mae']:.4f}
        MAPE: {metrics['mape']:.2f}%
        R² Score: {metrics['r2_score']:.4f}
        RUL Score: {metrics['rul_score']:.4f}
        
        Accuracy:
        ±10: {metrics['accuracy_10']:.1f}%
        ±20: {metrics['accuracy_20']:.1f}%
        ±30: {metrics['accuracy_30']:.1f}%
        
        Predictions:
        Early: {metrics['early_predictions']}
        Late: {metrics['late_predictions']}
        Total: {metrics['total_samples']}
        """
        ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Evaluation plots saved to {self.save_dir / 'evaluation_results.png'}")
    
    def _save_results(self, results: Dict[str, any]):
        """
        Save evaluation results to files
        
        Args:
            results: Evaluation results dictionary
        """
        # Save metrics to CSV
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(self.save_dir / 'evaluation_metrics.csv', index=False)
        
        # Save predictions and true values
        predictions_df = pd.DataFrame({
            'true_values': results['true_values'],
            'predictions': results['predictions'],
            'errors': results['predictions'] - results['true_values'],
            'absolute_errors': np.abs(results['predictions'] - results['true_values'])
        })
        predictions_df.to_csv(self.save_dir / 'predictions.csv', index=False)
        
        # Save detailed analysis if available
        if 'error_statistics' in results:
            error_stats_df = pd.DataFrame([results['error_statistics']])
            error_stats_df.to_csv(self.save_dir / 'error_statistics.csv', index=False)
            
        if 'range_analysis' in results:
            range_analysis_df = pd.DataFrame(results['range_analysis']).T
            range_analysis_df.to_csv(self.save_dir / 'range_analysis.csv')
        
        self.logger.info(f"Evaluation results saved to {self.save_dir}")
    
    def compare_models(self, model_results: Dict[str, Dict], 
                     save_comparison: bool = True) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            model_results: Dictionary of model names and their results
            save_comparison: Whether to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2_score'],
                'RUL Score': metrics['rul_score'],
                'Accuracy ±10': metrics['accuracy_10'],
                'Accuracy ±20': metrics['accuracy_20'],
                'Accuracy ±30': metrics['accuracy_30']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_comparison:
            comparison_df.to_csv(self.save_dir / 'model_comparison.csv', index=False)
            
            # Create comparison visualization
            self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """
        Create model comparison plots
        
        Args:
            comparison_df: DataFrame with model comparison data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'])
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # RUL Score comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['RUL Score'])
        axes[1, 1].set_title('RUL Score Comparison (Lower is Better)')
        axes[1, 1].set_ylabel('RUL Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()