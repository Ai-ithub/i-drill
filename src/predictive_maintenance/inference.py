#!/usr/bin/env python3
"""Inference module for generating wellbore images using trained StyleGAN2 model"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Union, Tuple
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.predictive_maintenance.config import (
    LATENT_DIM, IMAGE_SIZE, IMAGE_CHANNELS, DEVICE,
    GENERATOR_CONFIG, FAILURE_TYPES
)
from src.predictive_maintenance.gan.generator import StyleGAN2Generator
from src.predictive_maintenance.data.preprocessing import WellboreImagePreprocessor
from src.predictive_maintenance.utils import (
    load_checkpoint, save_image_grid, ensure_dir
)

class WellboreImageGenerator:
    """High-level interface for generating wellbore images"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config: Optional[dict] = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            config: Optional configuration dictionary
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.config = config or {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize generator
        self.generator = None
        self.preprocessor = None
        
        # Load model
        self.load_model()
        
        # Initialize preprocessor for post-processing
        self.preprocessor = WellboreImagePreprocessor(
            target_size=IMAGE_SIZE,
            normalize=True
        )
    
    def load_model(self):
        """Load trained generator model from checkpoint"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Create generator
        self.generator = StyleGAN2Generator(
            latent_dim=LATENT_DIM,
            image_size=IMAGE_SIZE,
            image_channels=IMAGE_CHANNELS,
            **GENERATOR_CONFIG
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            checkpoint = load_checkpoint(self.checkpoint_path)
            
            # Load generator state
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("No generator state found in checkpoint")
            
            self.logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Set to evaluation mode
        self.generator.eval()
    
    def generate_random_latent(self, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate random latent vectors"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        latent = torch.randn(batch_size, LATENT_DIM, device=self.device)
        return latent
    
    def generate_images(self, 
                       latent_vectors: Optional[torch.Tensor] = None,
                       num_images: int = 1,
                       seed: Optional[int] = None,
                       truncation_psi: float = 1.0) -> torch.Tensor:
        """Generate wellbore images
        
        Args:
            latent_vectors: Optional pre-defined latent vectors
            num_images: Number of images to generate (if latent_vectors not provided)
            seed: Random seed for reproducibility
            truncation_psi: Truncation parameter for style mixing (0-1)
        
        Returns:
            Generated images as tensor
        """
        with torch.no_grad():
            if latent_vectors is None:
                latent_vectors = self.generate_random_latent(num_images, seed)
            
            # Apply truncation if specified
            if truncation_psi < 1.0:
                # Calculate mean latent vector (would need to be pre-computed)
                # For now, just scale the latent vectors
                latent_vectors = latent_vectors * truncation_psi
            
            # Generate images
            generated_images = self.generator(latent_vectors)
            
            return generated_images
    
    def generate_interpolation(self, 
                             latent1: torch.Tensor,
                             latent2: torch.Tensor,
                             num_steps: int = 10) -> torch.Tensor:
        """Generate interpolation between two latent vectors"""
        interpolated_images = []
        
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
                
                generated_image = self.generator(interpolated_latent)
                interpolated_images.append(generated_image)
        
        return torch.cat(interpolated_images, dim=0)
    
    def generate_style_mixing(self, 
                            num_images: int = 4,
                            num_styles: int = 2) -> torch.Tensor:
        """Generate images with style mixing"""
        # Generate base latent vectors
        base_latents = self.generate_random_latent(num_images)
        style_latents = self.generate_random_latent(num_styles)
        
        mixed_images = []
        
        with torch.no_grad():
            for base_latent in base_latents:
                for style_latent in style_latents:
                    # Simple style mixing: use different latents for different layers
                    # This is a simplified version - full StyleGAN2 has more complex mixing
                    mixed_latent = torch.stack([base_latent, style_latent]).mean(dim=0, keepdim=True)
                    
                    generated_image = self.generator(mixed_latent)
                    mixed_images.append(generated_image)
        
        return torch.cat(mixed_images, dim=0)
    
    def generate_failure_type_samples(self, 
                                    failure_type: str,
                                    num_samples: int = 10,
                                    seed: Optional[int] = None) -> torch.Tensor:
        """Generate samples for specific failure type
        
        Note: This is a placeholder implementation. In practice, you would need
        conditional generation or fine-tuned models for specific failure types.
        """
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"Unknown failure type: {failure_type}. Available: {FAILURE_TYPES}")
        
        self.logger.info(f"Generating {num_samples} samples for failure type: {failure_type}")
        
        # For now, generate random samples
        # In practice, you might use conditional latent vectors or class-specific models
        generated_images = self.generate_images(num_images=num_samples, seed=seed)
        
        return generated_images
    
    def postprocess_images(self, images: torch.Tensor) -> List[np.ndarray]:
        """Convert generated tensors to displayable numpy arrays"""
        processed_images = []
        
        for image in images:
            # Use preprocessor's postprocess method
            processed_image = self.preprocessor.postprocess(image)
            processed_images.append(processed_image)
        
        return processed_images
    
    def save_images(self, 
                   images: torch.Tensor,
                   output_dir: str,
                   prefix: str = 'generated',
                   format: str = 'png') -> List[str]:
        """Save generated images to files"""
        ensure_dir(output_dir)
        
        processed_images = self.postprocess_images(images)
        saved_paths = []
        
        for i, image in enumerate(processed_images):
            filename = f"{prefix}_{i:04d}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Convert to PIL Image and save
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
            
            pil_image.save(filepath)
            saved_paths.append(filepath)
        
        self.logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def create_image_grid(self, 
                         images: torch.Tensor,
                         output_path: str,
                         nrow: int = 8,
                         title: Optional[str] = None) -> str:
        """Create and save an image grid"""
        save_image_grid(images, output_path, nrow=nrow)
        
        if title:
            # Add title to the saved image
            img = Image.open(output_path)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(img)
            ax.set_title(title, fontsize=16)
            ax.axis('off')
            
            # Save with title
            title_path = output_path.replace('.png', '_titled.png')
            plt.savefig(title_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return title_path
        
        return output_path

class WellboreInferencePipeline:
    """Complete inference pipeline for wellbore image generation"""
    
    def __init__(self, checkpoint_path: str, output_dir: str, config: Optional[dict] = None):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        ensure_dir(output_dir)
        
        # Initialize generator
        self.generator = WellboreImageGenerator(checkpoint_path, config=config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_sample_gallery(self, num_samples: int = 64, grid_size: int = 8):
        """Generate a gallery of sample images"""
        self.logger.info(f"Generating sample gallery with {num_samples} images")
        
        # Generate images
        images = self.generator.generate_images(num_images=num_samples)
        
        # Create grid
        grid_path = os.path.join(self.output_dir, 'sample_gallery.png')
        self.generator.create_image_grid(
            images, grid_path, nrow=grid_size, 
            title=f'Generated Wellbore Images ({num_samples} samples)'
        )
        
        # Save individual images
        individual_dir = os.path.join(self.output_dir, 'individual_samples')
        self.generator.save_images(images, individual_dir, prefix='sample')
        
        return grid_path
    
    def generate_interpolation_video(self, num_frames: int = 50, num_interpolations: int = 5):
        """Generate interpolation sequences"""
        self.logger.info(f"Generating {num_interpolations} interpolation sequences")
        
        interpolation_dir = os.path.join(self.output_dir, 'interpolations')
        ensure_dir(interpolation_dir)
        
        for i in range(num_interpolations):
            # Generate two random latent vectors
            latent1 = self.generator.generate_random_latent(1)
            latent2 = self.generator.generate_random_latent(1)
            
            # Generate interpolation
            interpolated_images = self.generator.generate_interpolation(
                latent1, latent2, num_frames
            )
            
            # Save as grid
            grid_path = os.path.join(interpolation_dir, f'interpolation_{i:02d}.png')
            self.generator.create_image_grid(
                interpolated_images, grid_path, nrow=num_frames//2,
                title=f'Latent Interpolation {i+1}'
            )
    
    def generate_failure_type_samples(self, samples_per_type: int = 20):
        """Generate samples for each failure type"""
        self.logger.info("Generating samples for each failure type")
        
        failure_dir = os.path.join(self.output_dir, 'failure_types')
        ensure_dir(failure_dir)
        
        for failure_type in FAILURE_TYPES:
            self.logger.info(f"Generating samples for {failure_type}")
            
            # Generate samples
            images = self.generator.generate_failure_type_samples(
                failure_type, samples_per_type
            )
            
            # Create type-specific directory
            type_dir = os.path.join(failure_dir, failure_type)
            ensure_dir(type_dir)
            
            # Save grid
            grid_path = os.path.join(type_dir, f'{failure_type}_grid.png')
            self.generator.create_image_grid(
                images, grid_path, nrow=5,
                title=f'{failure_type.replace("_", " ").title()} Samples'
            )
            
            # Save individual images
            self.generator.save_images(images, type_dir, prefix=failure_type)
    
    def generate_style_mixing_examples(self, num_examples: int = 16):
        """Generate style mixing examples"""
        self.logger.info("Generating style mixing examples")
        
        # Generate style mixed images
        mixed_images = self.generator.generate_style_mixing(num_examples//4, 4)
        
        # Save grid
        grid_path = os.path.join(self.output_dir, 'style_mixing.png')
        self.generator.create_image_grid(
            mixed_images, grid_path, nrow=4,
            title='Style Mixing Examples'
        )
        
        return grid_path
    
    def run_complete_inference(self):
        """Run complete inference pipeline"""
        self.logger.info("Running complete inference pipeline")
        
        # Generate sample gallery
        self.generate_sample_gallery()
        
        # Generate interpolations
        self.generate_interpolation_video()
        
        # Generate failure type samples
        self.generate_failure_type_samples()
        
        # Generate style mixing examples
        self.generate_style_mixing_examples()
        
        self.logger.info(f"Inference complete. Results saved to {self.output_dir}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate wellbore images using trained StyleGAN2')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated images')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--grid_size', type=int, default=8,
                       help='Grid size for image layout')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=DEVICE,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--failure_type', type=str, choices=FAILURE_TYPES,
                       help='Generate samples for specific failure type')
    parser.add_argument('--interpolation', action='store_true',
                       help='Generate interpolation sequences')
    parser.add_argument('--style_mixing', action='store_true',
                       help='Generate style mixing examples')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete inference pipeline')
    
    return parser.parse_args()

def main():
    """Main inference function"""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create inference pipeline
    pipeline = WellboreInferencePipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config={'device': args.device}
    )
    
    try:
        if args.complete:
            # Run complete pipeline
            pipeline.run_complete_inference()
        
        elif args.failure_type:
            # Generate samples for specific failure type
            pipeline.generator.generate_failure_type_samples(
                args.failure_type, args.num_samples
            )
        
        elif args.interpolation:
            # Generate interpolation sequences
            pipeline.generate_interpolation_video()
        
        elif args.style_mixing:
            # Generate style mixing examples
            pipeline.generate_style_mixing_examples()
        
        else:
            # Generate sample gallery
            pipeline.generate_sample_gallery(args.num_samples, args.grid_size)
        
        print(f"Generation complete! Results saved to {args.output_dir}")
    
    except Exception as e:
        print(f"Generation failed: {e}")
        raise

if __name__ == '__main__':
    main()