"""StyleGAN2 Training Pipeline for Wellbore Image Generation"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

from .generator import StyleGAN2Generator
from .discriminator import StyleGAN2Discriminator
from ..utils import (
    weights_init, save_image_grid, create_noise, gradient_penalty,
    plot_training_curves, ensure_dir, save_checkpoint, load_checkpoint
)
from ..config import GANConfig

class GANLoss:
    """Loss functions for StyleGAN2 training"""
    
    @staticmethod
    def adversarial_loss_g(fake_scores: torch.Tensor) -> torch.Tensor:
        """Generator adversarial loss (non-saturating)"""
        return F.softplus(-fake_scores).mean()
    
    @staticmethod
    def adversarial_loss_d(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        """Discriminator adversarial loss"""
        real_loss = F.softplus(-real_scores).mean()
        fake_loss = F.softplus(fake_scores).mean()
        return real_loss + fake_loss
    
    @staticmethod
    def r1_regularization(discriminator: nn.Module, real_images: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty for discriminator"""
        real_images.requires_grad_(True)
        real_scores = discriminator(real_images)
        
        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = (gradients.norm(2, dim=[1, 2, 3]) ** 2).mean()
        return gradient_penalty
    
    @staticmethod
    def path_length_regularization(generator: nn.Module, latent_codes: torch.Tensor,
                                 noise_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Path length regularization for generator"""
        # Generate images
        fake_images = generator(latent_codes, noise_inputs=noise_inputs)
        
        # Create noise for path length calculation
        noise = torch.randn_like(fake_images) / np.sqrt(np.prod(fake_images.shape[2:]))
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=(fake_images * noise).sum(),
            inputs=latent_codes,
            create_graph=True,
            retain_graph=True
        )[0]
        
        path_lengths = gradients.norm(2, dim=1)
        path_penalty = (path_lengths ** 2).mean()
        
        return path_penalty

class ExponentialMovingAverage:
    """Exponential moving average for generator weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update shadow weights"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow weights to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class GANTrainer:
    """StyleGAN2 Training Pipeline"""
    
    def __init__(self, config: GANConfig, dataloader: DataLoader, 
                 checkpoint_dir: str = "checkpoints", results_dir: str = "results"):
        self.config = config
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.device = config.DEVICE
        
        # Create directories
        ensure_dir(checkpoint_dir)
        ensure_dir(results_dir)
        
        # Initialize models
        self.generator = StyleGAN2Generator(
            latent_dim=config.LATENT_DIM,
            style_dim=config.LATENT_DIM,
            num_mapping_layers=config.G_MAPPING_LAYERS,
            num_synthesis_layers=config.G_SYNTHESIS_LAYERS,
            output_channels=config.IMAGE_CHANNELS,
            output_size=config.IMAGE_SIZE
        ).to(self.device)
        
        self.discriminator = StyleGAN2Discriminator(
            input_channels=config.IMAGE_CHANNELS,
            input_size=config.IMAGE_SIZE,
            num_layers=config.D_LAYERS
        ).to(self.device)
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(config.BETA1, config.BETA2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.BETA1, config.BETA2)
        )
        
        # Exponential moving average for generator
        self.ema = ExponentialMovingAverage(self.generator)
        
        # Loss function
        self.loss_fn = GANLoss()
        
        # Tensorboard writer
        self.writer = SummaryWriter(os.path.join(results_dir, "logs"))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.g_losses = []
        self.d_losses = []
        
        # Fixed noise for consistent sampling
        self.fixed_noise = create_noise(16, config.LATENT_DIM, self.device)
    
    def train_discriminator(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Train discriminator for one step"""
        self.optimizer_d.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Generate fake images
        noise = create_noise(batch_size, self.config.LATENT_DIM, self.device)
        noise_inputs = self.generator.generate_noise(batch_size, self.device)
        
        with torch.no_grad():
            fake_images = self.generator(noise, noise_inputs=noise_inputs)
        
        # Discriminator scores
        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)
        
        # Adversarial loss
        d_loss = self.loss_fn.adversarial_loss_d(real_scores, fake_scores)
        
        # R1 regularization (applied every 16 steps)
        r1_loss = 0
        if self.global_step % 16 == 0:
            r1_loss = self.loss_fn.r1_regularization(self.discriminator, real_images)
            r1_loss = r1_loss * self.config.R1_REGULARIZATION_WEIGHT
            d_loss = d_loss + r1_loss
        
        # Backward pass
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            "d_loss": d_loss.item(),
            "d_real_score": real_scores.mean().item(),
            "d_fake_score": fake_scores.mean().item(),
            "r1_loss": r1_loss.item() if isinstance(r1_loss, torch.Tensor) else r1_loss
        }
    
    def train_generator(self, batch_size: int) -> Dict[str, float]:
        """Train generator for one step"""
        self.optimizer_g.zero_grad()
        
        # Generate fake images
        noise = create_noise(batch_size, self.config.LATENT_DIM, self.device)
        noise_inputs = self.generator.generate_noise(batch_size, self.device)
        fake_images = self.generator(noise, noise_inputs=noise_inputs)
        
        # Discriminator scores
        fake_scores = self.discriminator(fake_images)
        
        # Adversarial loss
        g_loss = self.loss_fn.adversarial_loss_g(fake_scores)
        
        # Path length regularization (applied every 4 steps)
        pl_loss = 0
        if self.global_step % 4 == 0:
            pl_loss = self.loss_fn.path_length_regularization(
                self.generator, noise, noise_inputs
            )
            pl_loss = pl_loss * self.config.PATH_LENGTH_REGULARIZATION_WEIGHT
            g_loss = g_loss + pl_loss
        
        # Backward pass
        g_loss.backward()
        self.optimizer_g.step()
        
        # Update EMA
        self.ema.update(self.generator)
        
        return {
            "g_loss": g_loss.item(),
            "g_fake_score": fake_scores.mean().item(),
            "pl_loss": pl_loss.item() if isinstance(pl_loss, torch.Tensor) else pl_loss
        }
    
    def generate_samples(self, num_samples: int = 16, use_ema: bool = True) -> torch.Tensor:
        """Generate sample images"""
        self.generator.eval()
        
        if use_ema:
            self.ema.apply_shadow(self.generator)
        
        with torch.no_grad():
            noise = create_noise(num_samples, self.config.LATENT_DIM, self.device)
            noise_inputs = self.generator.generate_noise(num_samples, self.device)
            samples = self.generator(noise, noise_inputs=noise_inputs)
        
        if use_ema:
            self.ema.restore(self.generator)
        
        self.generator.train()
        return samples
    
    def save_samples(self, epoch: int, step: int):
        """Save generated samples"""
        samples = self.generate_samples()
        
        # Denormalize images (from [-1, 1] to [0, 1])
        samples = (samples + 1) / 2
        
        # Save image grid
        save_path = os.path.join(self.results_dir, f"samples_epoch_{epoch}_step_{step}.png")
        save_image_grid(samples, save_path, nrow=4)
    
    def log_metrics(self, epoch: int, step: int, d_metrics: Dict[str, float], 
                   g_metrics: Dict[str, float]):
        """Log training metrics"""
        # Tensorboard logging
        for key, value in d_metrics.items():
            self.writer.add_scalar(f"Discriminator/{key}", value, step)
        
        for key, value in g_metrics.items():
            self.writer.add_scalar(f"Generator/{key}", value, step)
        
        # Console logging
        if step % self.config.LOG_INTERVAL == 0:
            print(f"Epoch [{epoch}/{self.config.EPOCHS}] Step [{step}] "
                  f"D_Loss: {d_metrics['d_loss']:.4f} G_Loss: {g_metrics['g_loss']:.4f}")
    
    def save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'ema_shadow': self.ema.shadow,
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        self.ema.shadow = checkpoint['ema_shadow']
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Starting training on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training loop
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
            
            for batch_idx, (real_images, _) in enumerate(pbar):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Train discriminator
                d_metrics = self.train_discriminator(real_images)
                
                # Train generator
                g_metrics = self.train_generator(batch_size)
                
                # Update losses
                self.d_losses.append(d_metrics['d_loss'])
                self.g_losses.append(g_metrics['g_loss'])
                
                # Log metrics
                self.log_metrics(epoch, self.global_step, d_metrics, g_metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'D_Loss': f"{d_metrics['d_loss']:.4f}",
                    'G_Loss': f"{g_metrics['g_loss']:.4f}"
                })
                
                # Save samples
                if self.global_step % self.config.SAMPLE_INTERVAL == 0:
                    self.save_samples(epoch, self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.SAVE_INTERVAL == 0:
                    self.save_checkpoint(epoch, self.global_step)
                
                self.global_step += 1
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, self.global_step)
            
            # Plot training curves
            if (epoch + 1) % 10 == 0:
                plot_path = os.path.join(self.results_dir, f"training_curves_epoch_{epoch+1}.png")
                plot_training_curves(self.g_losses, self.d_losses, plot_path)
        
        print("Training completed!")
        self.writer.close()
    
    def evaluate(self, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.generator.eval()
        
        # Apply EMA weights
        self.ema.apply_shadow(self.generator)
        
        # Generate samples for evaluation
        all_samples = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                noise = create_noise(current_batch_size, self.config.LATENT_DIM, self.device)
                noise_inputs = self.generator.generate_noise(current_batch_size, self.device)
                samples = self.generator(noise, noise_inputs=noise_inputs)
                all_samples.append(samples.cpu())
        
        all_samples = torch.cat(all_samples, dim=0)
        
        # Restore original weights
        self.ema.restore(self.generator)
        self.generator.train()
        
        # Calculate evaluation metrics (placeholder)
        metrics = {
            "num_samples": len(all_samples),
            "sample_mean": all_samples.mean().item(),
            "sample_std": all_samples.std().item()
        }
        
        return metrics