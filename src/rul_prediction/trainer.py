#!/usr/bin/env python3
"""
Training Pipeline for RUL Prediction Models

This module provides training functionality for deep learning models
used in remaining useful life prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

class RULTrainer:
    """
    Trainer class for RUL prediction models
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine', patience: int = 10,
                 save_dir: str = './checkpoints'):
        """
        Initialize RUL Trainer
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cuda', 'cpu', or 'auto')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler_type: Type of learning rate scheduler
            patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
            
        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpoints
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # TensorBoard writer (optional)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.save_dir / 'tensorboard')
        else:
            self.writer = None
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data).squeeze()
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - targets))
                
            total_loss += loss.item() * data.size(0)
            total_mae += mae.item() * data.size(0)
            total_samples += data.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })
            
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_loss)
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data).squeeze()
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
                
                total_loss += loss.item() * data.size(0)
                total_mae += mae.item() * data.size(0)
                total_samples += data.size(0)
                
                # Store predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{mae.item():.4f}'
                })
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        
        # Calculate additional metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # RUL-specific scoring function
        rul_score = self._calculate_rul_score(predictions, targets)
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_loss),
            'rul_score': rul_score,
            'predictions': predictions,
            'targets': targets
        }
    
    def _calculate_rul_score(self, predictions: np.ndarray, 
                           targets: np.ndarray) -> float:
        """
        Calculate RUL-specific scoring function
        
        Args:
            predictions: Predicted RUL values
            targets: True RUL values
            
        Returns:
            RUL score (lower is better)
        """
        errors = predictions - targets
        
        # Asymmetric scoring: penalize late predictions more
        score = 0
        for error in errors:
            if error < 0:  # Early prediction
                score += np.exp(-error / 13) - 1
            else:  # Late prediction
                score += np.exp(error / 10) - 1
                
        return score / len(errors)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, save_best: bool = True,
              plot_progress: bool = True) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_best: Whether to save the best model
            plot_progress: Whether to plot training progress
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Log to TensorBoard (if available)
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
                self.writer.add_scalar('MAE/Train', train_metrics['mae'], epoch)
                self.writer.add_scalar('MAE/Validation', val_metrics['mae'], epoch)
                self.writer.add_scalar('RUL_Score/Validation', val_metrics['rul_score'], epoch)
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                
            # Log progress
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"RUL Score: {val_metrics['rul_score']:.4f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Plot training progress
        if plot_progress:
            self.plot_training_history()
            
        # Close TensorBoard writer (if available)
        if self.writer:
            self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            self.logger.info(f"Best model saved at epoch {epoch+1}")
            
        # Save regular checkpoint
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curves
        train_mae = [m['mae'] for m in self.train_metrics]
        val_mae = [m['mae'] for m in self.val_metrics]
        axes[0, 1].plot(train_mae, label='Train MAE')
        axes[0, 1].plot(val_mae, label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RUL Score
        rul_scores = [m['rul_score'] for m in self.val_metrics]
        axes[1, 0].plot(rul_scores, label='RUL Score', color='red')
        axes[1, 0].set_title('RUL Score (Lower is Better)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RUL Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Predictions vs Targets (last epoch)
        if self.val_metrics:
            last_predictions = self.val_metrics[-1]['predictions']
            last_targets = self.val_metrics[-1]['targets']
            axes[1, 1].scatter(last_targets, last_predictions, alpha=0.6)
            axes[1, 1].plot([last_targets.min(), last_targets.max()], 
                           [last_targets.min(), last_targets.max()], 'r--')
            axes[1, 1].set_title('Predictions vs Targets (Last Epoch)')
            axes[1, 1].set_xlabel('True RUL')
            axes[1, 1].set_ylabel('Predicted RUL')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training history plot saved to {self.save_dir / 'training_history.png'}")