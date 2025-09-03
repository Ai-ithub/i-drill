#!/usr/bin/env python3
"""
Basic Usage Examples for Wellbore Image Generation System

This script demonstrates how to use the GAN system programmatically
for training, inference, and evaluation.
"""

import os
import sys
import torch
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from config.inference_config import InferenceConfig
from train import GANTrainer
from inference import GANInference
from evaluation.metrics import EvaluationMetrics
from utils.file_utils import ensure_dir_exists


def example_training():
    """
    Example: Training a GAN model from scratch
    """
    print("=== Training Example ===")
    
    # Load training configuration
    config_path = "config/training_config_example.yaml"
    config = TrainingConfig.from_yaml(config_path)
    
    # Modify config for quick demo (optional)
    config.training.epochs = 10  # Reduce for demo
    config.training.batch_size = 4  # Reduce for demo
    config.data.train_dir = "data/train"  # Update paths as needed
    config.data.val_dir = "data/val"
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Start training
    print(f"Starting training for {config.training.epochs} epochs...")
    trainer.train()
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.best_model_path}")


def example_inference():
    """
    Example: Generating images using a trained model
    """
    print("\n=== Inference Example ===")
    
    # Load inference configuration
    config_path = "config/inference_config_example.yaml"
    config = InferenceConfig.from_yaml(config_path)
    
    # Update model path (use your trained model)
    config.model.model_path = "checkpoints/best_model.pth"
    config.generation.num_images = 10  # Generate 10 images
    config.generation.output_dir = "examples/generated"
    
    # Ensure output directory exists
    ensure_dir_exists(config.generation.output_dir)
    
    # Create inference engine
    inference = GANInference(config)
    
    # Load the trained model
    print(f"Loading model from: {config.model.model_path}")
    inference.load_model(config.model.model_path)
    
    # Generate images
    print(f"Generating {config.generation.num_images} images...")
    generated_images = inference.generate_batch(
        num_images=config.generation.num_images,
        batch_size=config.generation.batch_size
    )
    
    # Save images
    for i, image in enumerate(generated_images):
        output_path = os.path.join(
            config.generation.output_dir,
            f"generated_{i:04d}.{config.generation.image_format}"
        )
        inference.save_image(image, output_path)
    
    print(f"Images saved to: {config.generation.output_dir}")


def example_evaluation():
    """
    Example: Evaluating generated images
    """
    print("\n=== Evaluation Example ===")
    
    # Paths
    real_images_dir = "data/test"  # Directory with real images
    generated_images_dir = "examples/generated"  # Directory with generated images
    
    # Check if directories exist
    if not os.path.exists(real_images_dir):
        print(f"Real images directory not found: {real_images_dir}")
        return
    
    if not os.path.exists(generated_images_dir):
        print(f"Generated images directory not found: {generated_images_dir}")
        return
    
    # Create evaluation metrics calculator
    evaluator = EvaluationMetrics(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = evaluator.calculate_all_metrics(
        real_images_dir=real_images_dir,
        generated_images_dir=generated_images_dir,
        num_samples=100  # Use subset for demo
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"FID Score: {metrics['fid']:.4f}")
    print(f"Inception Score: {metrics['inception_score']:.4f} ± {metrics['inception_score_std']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"Diversity Score: {metrics['diversity']:.4f}")
    print(f"Mode Collapse Score: {metrics['mode_collapse']:.4f}")
    
    # Save metrics
    metrics_path = "examples/evaluation_results.json"
    evaluator.save_metrics(metrics, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")


def example_custom_generation():
    """
    Example: Custom image generation with specific parameters
    """
    print("\n=== Custom Generation Example ===")
    
    # Load configuration
    config = InferenceConfig.from_yaml("config/inference_config_example.yaml")
    config.model.model_path = "checkpoints/best_model.pth"
    
    # Create inference engine
    inference = GANInference(config)
    inference.load_model(config.model.model_path)
    
    # Generate with specific seed for reproducibility
    print("Generating images with specific seeds...")
    seeds = [42, 123, 456, 789, 999]
    
    for i, seed in enumerate(seeds):
        # Set seed
        torch.manual_seed(seed)
        
        # Generate single image
        image = inference.generate_single()
        
        # Save with seed in filename
        output_path = f"examples/generated/seed_{seed:04d}.png"
        inference.save_image(image, output_path)
        
        print(f"Generated image with seed {seed}: {output_path}")


def example_style_mixing():
    """
    Example: Style mixing between two latent codes
    """
    print("\n=== Style Mixing Example ===")
    
    # Load configuration
    config = InferenceConfig.from_yaml("config/inference_config_example.yaml")
    config.model.model_path = "checkpoints/best_model.pth"
    
    # Create inference engine
    inference = GANInference(config)
    inference.load_model(config.model.model_path)
    
    # Generate two base latent codes
    latent_dim = config.model.generator.latent_dim
    device = config.model.device
    
    latent1 = torch.randn(1, latent_dim, device=device)
    latent2 = torch.randn(1, latent_dim, device=device)
    
    # Create interpolation between latents
    num_steps = 5
    alphas = torch.linspace(0, 1, num_steps)
    
    print(f"Creating {num_steps} interpolated images...")
    
    for i, alpha in enumerate(alphas):
        # Interpolate latent codes
        mixed_latent = (1 - alpha) * latent1 + alpha * latent2
        
        # Generate image
        with torch.no_grad():
            image = inference.generator(mixed_latent)
        
        # Convert to PIL and save
        pil_image = inference.tensor_to_pil(image[0])
        output_path = f"examples/generated/interpolation_{i:02d}_alpha_{alpha:.2f}.png"
        pil_image.save(output_path)
        
        print(f"Saved interpolation step {i}: {output_path}")


def main():
    """
    Main function to run all examples
    """
    print("Wellbore Image Generation - Usage Examples")
    print("==========================================\n")
    
    # Create necessary directories
    os.makedirs("examples/generated", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available. Using CPU.")
    
    print("\nNote: Make sure you have:")
    print("1. Prepared your dataset in data/train, data/val, data/test")
    print("2. Trained a model or have a pretrained model checkpoint")
    print("3. Updated the configuration files with correct paths\n")
    
    # Run examples (comment out as needed)
    try:
        # Training example (requires dataset)
        # example_training()
        
        # Inference example (requires trained model)
        # example_inference()
        
        # Evaluation example (requires real and generated images)
        # example_evaluation()
        
        # Custom generation examples (requires trained model)
        # example_custom_generation()
        # example_style_mixing()
        
        print("\nExamples completed successfully!")
        print("Uncomment the example functions you want to run.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have:")
        print("- Proper dataset structure")
        print("- Trained model checkpoint")
        print("- Correct configuration files")


if __name__ == "__main__":
    main()