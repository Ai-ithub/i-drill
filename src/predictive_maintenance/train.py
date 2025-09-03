#!/usr/bin/env python3
"""Main training script for StyleGAN2-based wellbore image generation"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.predictive_maintenance.config import (
    LATENT_DIM, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, DEVICE, DATA_PATH, OUTPUT_PATH, CHECKPOINT_PATH,
    GENERATOR_CONFIG, DISCRIMINATOR_CONFIG, TRAINING_CONFIG
)
from src.predictive_maintenance.gan.generator import StyleGAN2Generator
from src.predictive_maintenance.gan.discriminator import StyleGAN2Discriminator
from src.predictive_maintenance.gan.trainer import GANTrainer
from src.predictive_maintenance.data.dataset import WellboreImageDataset
from src.predictive_maintenance.data.preprocessing import WellboreImagePreprocessor
from src.predictive_maintenance.data.augmentation import WellboreAugmentationPipeline
from src.predictive_maintenance.utils import (
    ensure_dir, save_checkpoint, load_checkpoint, plot_training_curves,
    calculate_fid_score, save_image_grid
)

class WellboreGANTrainingManager:
    """Main training manager for wellbore GAN model"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup logging
        self.setup_logging()
        
        # Create output directories
        self.setup_directories()
        
        # Initialize models
        self.generator = None
        self.discriminator = None
        self.trainer = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'fid_scores': [],
            'epochs': []
        }
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            self.init_wandb()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(OUTPUT_PATH, 'training.log'))
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training manager initialized with device: {self.device}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            OUTPUT_PATH,
            CHECKPOINT_PATH,
            os.path.join(OUTPUT_PATH, 'samples'),
            os.path.join(OUTPUT_PATH, 'plots'),
            os.path.join(OUTPUT_PATH, 'logs')
        ]
        
        for directory in directories:
            ensure_dir(directory)
            self.logger.info(f"Created directory: {directory}")
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.get('wandb_project', 'wellbore-gan'),
                name=self.config.get('experiment_name', f'stylegan2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=self.config
            )
            self.logger.info("Wandb initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
    
    def create_models(self):
        """Initialize generator and discriminator models"""
        self.logger.info("Creating StyleGAN2 models...")
        
        # Create generator
        self.generator = StyleGAN2Generator(
            latent_dim=LATENT_DIM,
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            **GENERATOR_CONFIG
        ).to(self.device)
        
        # Create discriminator
        self.discriminator = StyleGAN2Discriminator(
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            **DISCRIMINATOR_CONFIG
        ).to(self.device)
        
        # Log model information
        total_params_g = sum(p.numel() for p in self.generator.parameters())
        total_params_d = sum(p.numel() for p in self.discriminator.parameters())
        
        self.logger.info(f"Generator parameters: {total_params_g:,}")
        self.logger.info(f"Discriminator parameters: {total_params_d:,}")
        self.logger.info(f"Total parameters: {total_params_g + total_params_d:,}")
    
    def create_datasets(self):
        """Create training and validation datasets"""
        self.logger.info("Creating datasets...")
        
        # Initialize preprocessor
        preprocessor = WellboreImagePreprocessor(
            target_size=IMAGE_SIZE,
            normalize=True,
            enhance_contrast=True,
            denoise=True
        )
        
        # Initialize augmentation pipeline
        augmentation_pipeline = WellboreAugmentationPipeline(
            train_augmentations=True,
            validation_augmentations=False
        )
        
        # Create training dataset
        train_dataset = WellboreImageDataset(
            data_path=DATA_PATH,
            split='train',
            preprocessor=preprocessor,
            augmentation=augmentation_pipeline.get_train_transform(),
            failure_types=self.config.get('failure_types', None)
        )
        
        # Create validation dataset
        val_dataset = WellboreImageDataset(
            data_path=DATA_PATH,
            split='val',
            preprocessor=preprocessor,
            augmentation=augmentation_pipeline.get_val_transform(),
            failure_types=self.config.get('failure_types', None)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Training batches: {len(self.train_loader)}")
        self.logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def create_trainer(self):
        """Initialize the GAN trainer"""
        self.logger.info("Creating GAN trainer...")
        
        self.trainer = GANTrainer(
            generator=self.generator,
            discriminator=self.discriminator,
            device=self.device,
            **TRAINING_CONFIG
        )
        
        self.logger.info("GAN trainer created successfully")
    
    def load_checkpoint_if_exists(self):
        """Load checkpoint if it exists"""
        checkpoint_file = os.path.join(CHECKPOINT_PATH, 'latest_checkpoint.pth')
        
        if os.path.exists(checkpoint_file):
            self.logger.info(f"Loading checkpoint from {checkpoint_file}")
            
            checkpoint = load_checkpoint(checkpoint_file)
            
            # Load model states
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Load trainer state
            self.trainer.load_state(checkpoint)
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            self.logger.info(f"Resumed from epoch {self.current_epoch}")
        else:
            self.logger.info("No checkpoint found, starting from scratch")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Add trainer state
        trainer_state = self.trainer.get_state()
        checkpoint.update(trainer_state)
        
        # Save latest checkpoint
        checkpoint_file = os.path.join(CHECKPOINT_PATH, 'latest_checkpoint.pth')
        save_checkpoint(checkpoint, checkpoint_file)
        
        # Save best checkpoint if applicable
        if is_best:
            best_checkpoint_file = os.path.join(CHECKPOINT_PATH, 'best_checkpoint.pth')
            save_checkpoint(checkpoint, best_checkpoint_file)
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def generate_samples(self, epoch: int, num_samples: int = 64):
        """Generate and save sample images"""
        self.logger.info(f"Generating {num_samples} samples at epoch {epoch}")
        
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, LATENT_DIM, device=self.device)
            
            # Generate images
            fake_images = self.generator(noise)
            
            # Save image grid
            output_file = os.path.join(OUTPUT_PATH, 'samples', f'epoch_{epoch:04d}.png')
            save_image_grid(fake_images, output_file, nrow=8)
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                try:
                    wandb.log({
                        'generated_samples': wandb.Image(output_file),
                        'epoch': epoch
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log to wandb: {e}")
    
    def evaluate_model(self, epoch: int):
        """Evaluate model performance"""
        self.logger.info(f"Evaluating model at epoch {epoch}")
        
        # Calculate FID score
        try:
            fid_score = self.calculate_fid_score()
            self.training_history['fid_scores'].append(fid_score)
            
            self.logger.info(f"FID Score at epoch {epoch}: {fid_score:.4f}")
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                try:
                    wandb.log({
                        'fid_score': fid_score,
                        'epoch': epoch
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log to wandb: {e}")
            
            return fid_score
        
        except Exception as e:
            self.logger.warning(f"Failed to calculate FID score: {e}")
            return float('inf')
    
    def calculate_fid_score(self, num_samples: int = 1000):
        """Calculate FID score between real and generated images"""
        # Generate fake images
        fake_images = []
        
        with torch.no_grad():
            for _ in range(0, num_samples, BATCH_SIZE):
                batch_size = min(BATCH_SIZE, num_samples - len(fake_images))
                noise = torch.randn(batch_size, LATENT_DIM, device=self.device)
                fake_batch = self.generator(noise)
                fake_images.extend(fake_batch.cpu())
        
        fake_images = torch.stack(fake_images[:num_samples])
        
        # Get real images
        real_images = []
        for batch_idx, (real_batch, _) in enumerate(self.val_loader):
            real_images.extend(real_batch)
            if len(real_images) >= num_samples:
                break
        
        real_images = torch.stack(real_images[:num_samples])
        
        # Calculate FID
        fid_score = calculate_fid_score(real_images, fake_images, device=self.device)
        return fid_score
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.logger.info(f"Training epoch {epoch}/{NUM_EPOCHS}")
        
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Training loop
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            
            # Train discriminator and generator
            d_loss, g_loss = self.trainer.train_step(real_images)
            
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss:.4f}',
                'G_loss': f'{g_loss:.4f}'
            })
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                try:
                    wandb.log({
                        'batch_d_loss': d_loss,
                        'batch_g_loss': g_loss,
                        'batch': epoch * num_batches + batch_idx
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log to wandb: {e}")
        
        # Calculate average losses
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        
        # Update training history
        self.training_history['discriminator_loss'].append(avg_d_loss)
        self.training_history['generator_loss'].append(avg_g_loss)
        self.training_history['epochs'].append(epoch)
        
        self.logger.info(f"Epoch {epoch} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        return avg_d_loss, avg_g_loss
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        best_fid = float('inf')
        
        for epoch in range(self.current_epoch + 1, NUM_EPOCHS + 1):
            # Train for one epoch
            d_loss, g_loss = self.train_epoch(epoch)
            
            # Generate samples
            if epoch % self.config.get('sample_interval', 10) == 0:
                self.generate_samples(epoch)
            
            # Evaluate model
            if epoch % self.config.get('eval_interval', 20) == 0:
                fid_score = self.evaluate_model(epoch)
                
                # Check if this is the best model
                is_best = fid_score < best_fid
                if is_best:
                    best_fid = fid_score
                    self.logger.info(f"New best FID score: {best_fid:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                # Save regular checkpoint
                if epoch % self.config.get('checkpoint_interval', 50) == 0:
                    self.save_checkpoint(epoch)
            
            # Plot training curves
            if epoch % self.config.get('plot_interval', 50) == 0:
                plot_file = os.path.join(OUTPUT_PATH, 'plots', f'training_curves_epoch_{epoch}.png')
                plot_training_curves(self.training_history, plot_file)
        
        self.logger.info("Training completed!")
        
        # Final evaluation and sample generation
        self.generate_samples(NUM_EPOCHS, num_samples=100)
        final_fid = self.evaluate_model(NUM_EPOCHS)
        
        # Save final checkpoint
        self.save_checkpoint(NUM_EPOCHS, is_best=final_fid < best_fid)
        
        # Plot final training curves
        final_plot_file = os.path.join(OUTPUT_PATH, 'plots', 'final_training_curves.png')
        plot_training_curves(self.training_history, final_plot_file)
        
        self.logger.info(f"Final FID score: {final_fid:.4f}")
        self.logger.info(f"Best FID score: {best_fid:.4f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train StyleGAN2 for wellbore image generation')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--output_path', type=str, help='Path to save outputs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    default_config = {
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'data_path': DATA_PATH,
        'output_path': OUTPUT_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
        'num_workers': 4,
        'sample_interval': 10,
        'eval_interval': 20,
        'checkpoint_interval': 50,
        'plot_interval': 50,
        'use_wandb': False,
        'wandb_project': 'wellbore-gan',
        'experiment_name': f'stylegan2_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_level': 'INFO',
        'failure_types': None  # Train on all failure types
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        default_config.update(file_config)
    
    return default_config

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_path:
        config['output_path'] = args.output_path
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    if args.wandb:
        config['use_wandb'] = True
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    # Create training manager
    training_manager = WellboreGANTrainingManager(config)
    
    try:
        # Initialize everything
        training_manager.create_models()
        training_manager.create_datasets()
        training_manager.create_trainer()
        
        # Load checkpoint if resuming
        if args.resume:
            training_manager.load_checkpoint_if_exists()
        
        # Start training
        training_manager.train()
        
    except KeyboardInterrupt:
        training_manager.logger.info("Training interrupted by user")
        # Save checkpoint before exiting
        training_manager.save_checkpoint(training_manager.current_epoch)
        
    except Exception as e:
        training_manager.logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if config.get('use_wandb', False):
            try:
                wandb.finish()
            except:
                pass

if __name__ == '__main__':
    main()