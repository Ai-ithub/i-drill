#!/usr/bin/env python3
"""Training script for StyleGAN2 wellbore image generation"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from config import GANConfig
from gan.generator import StyleGAN2Generator
from gan.discriminator import StyleGAN2Discriminator
from gan.trainer import StyleGAN2Trainer
from data import WellboreImageDataset
from utils import (
    save_checkpoint, load_checkpoint, ensure_dir,
    AverageMeter, ProgressMeter, format_time
)

def create_data_loader(config: GANConfig) -> DataLoader:
    """Create data loader for training"""
    logging.info(f"Loading dataset from: {config.DATA_PATH}")
    
    dataset = WellboreImageDataset(
        data_path=config.DATA_PATH,
        image_size=config.IMAGE_SIZE,
        augment=config.USE_AUGMENTATION
    )
    
    logging.info(f"Dataset size: {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def create_models(config: GANConfig) -> tuple:
    """Create generator and discriminator models"""
    logging.info("Creating StyleGAN2 models...")
    
    # Create generator
    generator = StyleGAN2Generator(
        latent_dim=config.LATENT_DIM,
        image_size=config.IMAGE_SIZE,
        num_channels=config.NUM_CHANNELS,
        num_layers=config.NUM_LAYERS,
        feature_maps=config.FEATURE_MAPS
    )
    
    # Create discriminator
    discriminator = StyleGAN2Discriminator(
        image_size=config.IMAGE_SIZE,
        num_channels=config.NUM_CHANNELS,
        num_layers=config.NUM_LAYERS,
        feature_maps=config.FEATURE_MAPS
    )
    
    # Move to device
    generator = generator.to(config.DEVICE)
    discriminator = discriminator.to(config.DEVICE)
    
    # Log model parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    logging.info(f"Generator parameters: {gen_params:,}")
    logging.info(f"Discriminator parameters: {disc_params:,}")
    logging.info(f"Total parameters: {gen_params + disc_params:,}")
    
    return generator, discriminator

def setup_training(config: GANConfig, generator: nn.Module, discriminator: nn.Module) -> StyleGAN2Trainer:
    """Setup trainer with optimizers and schedulers"""
    logging.info("Setting up training components...")
    
    trainer = StyleGAN2Trainer(
        generator=generator,
        discriminator=discriminator,
        config=config
    )
    
    return trainer

def save_sample_images(generator: nn.Module, config: GANConfig, epoch: int, output_dir: str):
    """Generate and save sample images"""
    generator.eval()
    
    with torch.no_grad():
        # Generate fixed noise for consistent samples
        fixed_noise = torch.randn(16, config.LATENT_DIM, device=config.DEVICE)
        fake_images = generator(fixed_noise)
        
        # Denormalize images (from [-1, 1] to [0, 1])
        fake_images = (fake_images + 1) / 2
        
        # Create grid and save
        grid = make_grid(fake_images, nrow=4, normalize=True)
        save_path = os.path.join(output_dir, 'samples', f'epoch_{epoch:04d}.png')
        save_image(grid, save_path)
    
    generator.train()

def train_epoch(trainer: StyleGAN2Trainer, dataloader: DataLoader, epoch: int, 
                config: GANConfig, writer: SummaryWriter) -> dict:
    """Train for one epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    g_losses = AverageMeter('G_Loss', ':.4f')
    d_losses = AverageMeter('D_Loss', ':.4f')
    
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, g_losses, d_losses],
        prefix=f"Epoch: [{epoch}]"
    )
    
    trainer.generator.train()
    trainer.discriminator.train()
    
    end = time.time()
    
    for i, real_images in enumerate(dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move to device
        real_images = real_images.to(config.DEVICE)
        
        # Train discriminator
        d_loss = trainer.train_discriminator(real_images)
        
        # Train generator (every G_STEPS iterations)
        g_loss = 0.0
        if i % config.G_STEPS == 0:
            g_loss = trainer.train_generator(real_images.size(0))
        
        # Update meters
        g_losses.update(g_loss, real_images.size(0))
        d_losses.update(d_loss, real_images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', g_loss, global_step)
        writer.add_scalar('Loss/Discriminator', d_loss, global_step)
        
        # Print progress
        if i % config.PRINT_FREQ == 0:
            progress.display(i)
    
    return {
        'g_loss': g_losses.avg,
        'd_loss': d_losses.avg,
        'batch_time': batch_time.avg
    }

def train_gan(config: GANConfig, output_dir: str):
    """Main training function"""
    logging.info("Starting StyleGAN2 training...")
    
    # Create output directories
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    sample_dir = os.path.join(output_dir, 'samples')
    log_dir = os.path.join(output_dir, 'logs')
    
    ensure_dir(checkpoint_dir)
    ensure_dir(sample_dir)
    ensure_dir(log_dir)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create data loader
    dataloader = create_data_loader(config)
    
    # Create models
    generator, discriminator = create_models(config)
    
    # Setup training
    trainer = setup_training(config, generator, discriminator)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if config.RESUME_CHECKPOINT:
        if os.path.exists(config.RESUME_CHECKPOINT):
            logging.info(f"Resuming from checkpoint: {config.RESUME_CHECKPOINT}")
            checkpoint = load_checkpoint(config.RESUME_CHECKPOINT)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            trainer.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed from epoch {start_epoch}")
        else:
            logging.warning(f"Checkpoint not found: {config.RESUME_CHECKPOINT}")
    
    # Training loop
    logging.info(f"Training for {config.NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train for one epoch
        metrics = train_epoch(trainer, dataloader, epoch, config, writer)
        
        # Update learning rate
        trainer.update_learning_rate()
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start
        logging.info(
            f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
            f"G_Loss: {metrics['g_loss']:.4f} "
            f"D_Loss: {metrics['d_loss']:.4f} "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Save sample images
        if epoch % config.SAMPLE_FREQ == 0:
            save_sample_images(generator, config, epoch, output_dir)
        
        # Save checkpoint
        if epoch % config.CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            save_checkpoint({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': trainer.g_optimizer.state_dict(),
                'd_optimizer': trainer.d_optimizer.state_dict(),
                'config': config.__dict__
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = os.path.join(checkpoint_dir, 'final_model.pth')
    save_checkpoint({
        'epoch': config.NUM_EPOCHS - 1,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': trainer.g_optimizer.state_dict(),
        'd_optimizer': trainer.d_optimizer.state_dict(),
        'config': config.__dict__
    }, final_checkpoint)
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {format_time(total_time)}")
    logging.info(f"Final model saved: {final_checkpoint}")
    
    writer.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train StyleGAN2 for wellbore images')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = GANConfig.from_yaml(args.config)
    
    # Start training
    train_gan(config, args.output_dir)