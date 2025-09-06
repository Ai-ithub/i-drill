#!/usr/bin/env python3
"""Inference script for generating wellbore images using trained StyleGAN2"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
from PIL import Image

from config import GANConfig
from gan.generator import StyleGAN2Generator
from utils import load_checkpoint, ensure_dir, format_time

class ImageGenerator:
    """Wrapper class for image generation using trained StyleGAN2"""
    
    def __init__(self, config: GANConfig, checkpoint_path: str):
        self.config = config
        self.device = config.DEVICE
        
        # Load generator
        self.generator = self._load_generator(checkpoint_path)
        self.generator.eval()
        
        logging.info(f"Generator loaded from: {checkpoint_path}")
        logging.info(f"Using device: {self.device}")
    
    def _load_generator(self, checkpoint_path: str) -> nn.Module:
        """Load generator from checkpoint"""
        # Create generator
        generator = StyleGAN2Generator(
            latent_dim=self.config.LATENT_DIM,
            image_size=self.config.IMAGE_SIZE,
            num_channels=self.config.NUM_CHANNELS,
            num_layers=self.config.NUM_LAYERS,
            feature_maps=self.config.FEATURE_MAPS
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        generator.load_state_dict(checkpoint['generator'])
        generator = generator.to(self.device)
        
        return generator
    
    def generate_random_images(self, num_images: int, batch_size: int = 16) -> List[torch.Tensor]:
        """Generate random images"""
        logging.info(f"Generating {num_images} random images...")
        
        generated_images = []
        num_batches = (num_images + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(batch_size, num_images - i * batch_size)
                
                # Generate random latent vectors
                latent_vectors = torch.randn(
                    current_batch_size, 
                    self.config.LATENT_DIM, 
                    device=self.device
                )
                
                # Generate images
                fake_images = self.generator(latent_vectors)
                
                # Denormalize from [-1, 1] to [0, 1]
                fake_images = (fake_images + 1) / 2
                fake_images = torch.clamp(fake_images, 0, 1)
                
                generated_images.append(fake_images.cpu())
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Generated batch {i + 1}/{num_batches}")
        
        return generated_images
    
    def generate_from_latent(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """Generate images from specific latent vectors"""
        with torch.no_grad():
            latent_vectors = latent_vectors.to(self.device)
            fake_images = self.generator(latent_vectors)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            fake_images = torch.clamp(fake_images, 0, 1)
            
            return fake_images.cpu()
    
    def interpolate_latent(self, latent1: torch.Tensor, latent2: torch.Tensor, 
                          num_steps: int = 10) -> List[torch.Tensor]:
        """Interpolate between two latent vectors"""
        interpolated_images = []
        
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
                
                fake_image = self.generator(interpolated_latent.to(self.device))
                fake_image = (fake_image + 1) / 2
                fake_image = torch.clamp(fake_image, 0, 1)
                
                interpolated_images.append(fake_image.cpu())
        
        return interpolated_images
    
    def generate_style_mixing(self, num_images: int = 16) -> torch.Tensor:
        """Generate images with style mixing"""
        with torch.no_grad():
            # Generate two sets of latent vectors
            latent1 = torch.randn(num_images, self.config.LATENT_DIM, device=self.device)
            latent2 = torch.randn(num_images, self.config.LATENT_DIM, device=self.device)
            
            # Mix styles at different layers
            mixed_images = self.generator.style_mixing(latent1, latent2)
            
            # Denormalize
            mixed_images = (mixed_images + 1) / 2
            mixed_images = torch.clamp(mixed_images, 0, 1)
            
            return mixed_images.cpu()

def save_images_individually(images: List[torch.Tensor], output_dir: str, 
                           prefix: str = 'generated') -> List[str]:
    """Save images individually"""
    ensure_dir(output_dir)
    saved_paths = []
    
    image_idx = 0
    for batch in images:
        for i in range(batch.size(0)):
            image_path = os.path.join(output_dir, f'{prefix}_{image_idx:06d}.png')
            save_image(batch[i], image_path)
            saved_paths.append(image_path)
            image_idx += 1
    
    return saved_paths

def save_image_grid(images: List[torch.Tensor], output_path: str, 
                   nrow: int = 8) -> None:
    """Save images as a grid"""
    # Concatenate all batches
    all_images = torch.cat(images, dim=0)
    
    # Create grid
    grid = make_grid(all_images, nrow=nrow, normalize=True, padding=2)
    
    # Save grid
    save_image(grid, output_path)

def create_interpolation_video(generator: ImageGenerator, output_path: str,
                             num_frames: int = 100, fps: int = 30) -> None:
    """Create interpolation video between random latent vectors"""
    try:
        import cv2
    except ImportError:
        logging.warning("OpenCV not available, skipping video generation")
        return
    
    logging.info(f"Creating interpolation video with {num_frames} frames...")
    
    # Generate random start and end points
    latent1 = torch.randn(1, generator.config.LATENT_DIM)
    latent2 = torch.randn(1, generator.config.LATENT_DIM)
    
    # Generate interpolated images
    interpolated_images = generator.interpolate_latent(latent1, latent2, num_frames)
    
    # Setup video writer
    height, width = generator.config.IMAGE_SIZE, generator.config.IMAGE_SIZE
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for image_tensor in interpolated_images:
        # Convert tensor to numpy array
        image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        video_writer.write(image_np)
    
    video_writer.release()
    logging.info(f"Video saved: {output_path}")

def generate_images(config: GANConfig, checkpoint_path: str, output_dir: str,
                   num_images: int = 100, batch_size: int = 16,
                   save_individual: bool = True, save_grid: bool = True,
                   create_video: bool = False) -> None:
    """Main image generation function"""
    logging.info("Starting image generation...")
    start_time = time.time()
    
    # Create output directories
    individual_dir = os.path.join(output_dir, 'individual')
    grid_dir = os.path.join(output_dir, 'grids')
    video_dir = os.path.join(output_dir, 'videos')
    
    if save_individual:
        ensure_dir(individual_dir)
    if save_grid:
        ensure_dir(grid_dir)
    if create_video:
        ensure_dir(video_dir)
    
    # Initialize generator
    generator = ImageGenerator(config, checkpoint_path)
    
    # Generate random images
    generated_images = generator.generate_random_images(num_images, batch_size)
    
    # Save individual images
    if save_individual:
        logging.info("Saving individual images...")
        saved_paths = save_images_individually(generated_images, individual_dir)
        logging.info(f"Saved {len(saved_paths)} individual images to: {individual_dir}")
    
    # Save image grid
    if save_grid:
        logging.info("Creating image grid...")
        grid_path = os.path.join(grid_dir, f'generated_grid_{num_images}.png')
        save_image_grid(generated_images, grid_path)
        logging.info(f"Image grid saved: {grid_path}")
    
    # Generate style mixing examples
    logging.info("Generating style mixing examples...")
    mixed_images = generator.generate_style_mixing(16)
    mixed_grid_path = os.path.join(grid_dir, 'style_mixing_examples.png')
    grid = make_grid(mixed_images, nrow=4, normalize=True, padding=2)
    save_image(grid, mixed_grid_path)
    logging.info(f"Style mixing examples saved: {mixed_grid_path}")
    
    # Create interpolation video
    if create_video:
        logging.info("Creating interpolation video...")
        video_path = os.path.join(video_dir, 'latent_interpolation.mp4')
        create_interpolation_video(generator, video_path)
    
    total_time = time.time() - start_time
    logging.info(f"Image generation completed in {format_time(total_time)}")
    logging.info(f"Results saved to: {output_dir}")

def generate_specific_samples(config: GANConfig, checkpoint_path: str, 
                            output_dir: str, seed: int = 42) -> None:
    """Generate specific samples with fixed seed for reproducibility"""
    logging.info(f"Generating reproducible samples with seed: {seed}")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize generator
    generator = ImageGenerator(config, checkpoint_path)
    
    # Generate fixed samples
    num_samples = 64
    latent_vectors = torch.randn(num_samples, config.LATENT_DIM)
    generated_images = generator.generate_from_latent(latent_vectors)
    
    # Save as grid
    grid_path = os.path.join(output_dir, f'fixed_samples_seed_{seed}.png')
    grid = make_grid(generated_images, nrow=8, normalize=True, padding=2)
    save_image(grid, grid_path)
    
    logging.info(f"Fixed samples saved: {grid_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate images using trained StyleGAN2')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='generated_images', help='Output directory')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--create-video', action='store_true', help='Create interpolation video')
    
    args = parser.parse_args()
    
    # Load config
    config = GANConfig.from_yaml(args.config)
    
    # Generate images
    generate_images(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        create_video=args.create_video
    )