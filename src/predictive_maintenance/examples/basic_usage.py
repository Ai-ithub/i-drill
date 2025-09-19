#!/usr/bin/env python3
"""Basic usage examples for wellbore image generation with StyleGAN2"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import GANConfig
from gan import StyleGAN2Generator, StyleGAN2Discriminator
from train import train_model
from inference import ImageGenerator, generate_images
from evaluation import ModelEvaluator, evaluate_model
from data import WellboreImageDataset, create_dataloader
from utils import setup_logging, get_device, ensure_dir

def example_training():
    """Example: Train a StyleGAN2 model from scratch"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Training StyleGAN2 Model")
    print("="*50)
    
    # Create configuration
    config = GANConfig(
        # Data settings
        data_path="data/wellbore_images",
        image_size=256,
        batch_size=16,
        
        # Model settings
        latent_dim=512,
        num_mapping_layers=8,
        
        # Training settings
        num_epochs=100,
        learning_rate=0.002,
        beta1=0.0,
        beta2=0.99,
        
        # Output settings
        output_dir="outputs/training",
        checkpoint_dir="checkpoints",
        sample_dir="samples",
        
        # Logging
        log_interval=100,
        save_interval=1000,
        sample_interval=500
    )
    
    print(f"Configuration created:")
    print(f"  Data path: {config.data_path}")
    print(f"  Image size: {config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Number of epochs: {config.num_epochs}")
    
    # Check if data exists
    if not os.path.exists(config.data_path):
        print(f"\nWarning: Data path '{config.data_path}' does not exist.")
        print("Please prepare your dataset first using data_preparation.py")
        print("Example: python data_preparation.py --action sample --output-dir data/wellbore_images")
        return
    
    # Start training
    print(f"\nStarting training...")
    print(f"Output directory: {config.output_dir}")
    
    try:
        # Train the model
        train_model(config)
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print("Please check your data and configuration.")

def example_inference():
    """Example: Generate images using a trained model"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Image Generation")
    print("="*50)
    
    # Configuration for inference
    model_path = "checkpoints/stylegan2_generator_latest.pth"
    output_dir = "outputs/generated_images"
    num_images = 16
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {num_images}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nWarning: Model file '{model_path}' does not exist.")
        print("Please train a model first using example_training() or:")
        print("python main.py --mode train --config configs/default.yaml")
        return
    
    try:
        # Create configuration (minimal for inference)
        config = GANConfig(
            latent_dim=512,
            image_size=256,
            num_mapping_layers=8
        )
        
        # Generate images
        print("\nGenerating images...")
        
        generate_images(
            config=config,
            model_path=model_path,
            output_dir=output_dir,
            num_images=num_images
        )
        
        print(f"\nImages generated successfully!")
        print(f"Check the output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nImage generation failed: {str(e)}")

def example_custom_generation():
    """Example: Custom image generation with specific latent codes"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Custom Image Generation")
    print("="*50)
    
    model_path = "checkpoints/stylegan2_generator_latest.pth"
    output_dir = "outputs/custom_generation"
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' does not exist. Please train a model first.")
        return
    
    try:
        import torch
        import numpy as np
        
        # Create configuration
        config = GANConfig(
            latent_dim=512,
            image_size=256,
            num_mapping_layers=8
        )
        
        # Initialize generator
        print("Loading generator...")
        generator = ImageGenerator(config, model_path)
        
        ensure_dir(output_dir)
        
        # Example 1: Generate from random latent codes
        print("\n1. Generating from random latent codes...")
        random_images = generator.generate_random(num_images=4)
        generator.save_images(random_images, output_dir, prefix="random")
        
        # Example 2: Generate from specific latent codes
        print("2. Generating from specific latent codes...")
        
        # Create specific latent codes
        device = get_device()
        latent_codes = []
        
        # Smooth latent code (low values)
        smooth_latent = torch.randn(1, config.latent_dim, device=device) * 0.5
        latent_codes.append(smooth_latent)
        
        # Sharp latent code (high values)
        sharp_latent = torch.randn(1, config.latent_dim, device=device) * 2.0
        latent_codes.append(sharp_latent)
        
        # Mixed latent code
        mixed_latent = torch.cat([
            torch.randn(1, config.latent_dim // 2, device=device) * 0.5,
            torch.randn(1, config.latent_dim // 2, device=device) * 1.5
        ], dim=1)
        latent_codes.append(mixed_latent)
        
        specific_images = generator.generate_from_latents(latent_codes)
        generator.save_images(specific_images, output_dir, prefix="specific")
        
        # Example 3: Style mixing
        print("3. Generating with style mixing...")
        
        # Generate two base latent codes
        latent1 = torch.randn(1, config.latent_dim, device=device)
        latent2 = torch.randn(1, config.latent_dim, device=device)
        
        # Create style mixing at different layers
        mixed_images = generator.generate_style_mixing(
            latent1, latent2, 
            mixing_layers=[4, 8, 12]  # Mix at these layers
        )
        generator.save_images(mixed_images, output_dir, prefix="style_mixed")
        
        # Example 4: Interpolation
        print("4. Generating interpolation sequence...")
        
        interpolated_images = generator.generate_interpolation(
            latent1, latent2, steps=8
        )
        generator.save_images(interpolated_images, output_dir, prefix="interpolation")
        
        # Create interpolation video
        generator.create_interpolation_video(
            latent1, latent2,
            output_path=os.path.join(output_dir, "interpolation.mp4"),
            steps=30, fps=10
        )
        
        print(f"\nCustom generation completed!")
        print(f"Check the output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nCustom generation failed: {str(e)}")

def example_evaluation():
    """Example: Evaluate model quality"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Model Evaluation")
    print("="*50)
    
    model_path = "checkpoints/stylegan2_generator_latest.pth"
    real_data_path = "data/wellbore_images"
    output_dir = "outputs/evaluation"
    
    print(f"Model path: {model_path}")
    print(f"Real data path: {real_data_path}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' does not exist. Please train a model first.")
        return
    
    if not os.path.exists(real_data_path):
        print(f"Real data path '{real_data_path}' does not exist.")
        return
    
    try:
        # Create configuration
        config = GANConfig(
            latent_dim=512,
            image_size=256,
            num_mapping_layers=8,
            batch_size=16
        )
        
        print("\nStarting evaluation...")
        
        # Evaluate model
        results = evaluate_model(
            config=config,
            model_path=model_path,
            real_data_path=real_data_path,
            output_dir=output_dir,
            num_samples=1000  # Number of samples for evaluation
        )
        
        print("\nEvaluation Results:")
        print(f"FID Score: {results['fid']:.4f}")
        print(f"IS Score: {results['is_mean']:.4f} ± {results['is_std']:.4f}")
        print(f"LPIPS Score: {results['lpips']:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        if results['fid'] < 50:
            print("✓ FID: Excellent quality (< 50)")
        elif results['fid'] < 100:
            print("○ FID: Good quality (50-100)")
        else:
            print("✗ FID: Needs improvement (> 100)")
        
        if results['is_mean'] > 3:
            print("✓ IS: Good diversity and quality (> 3)")
        else:
            print("○ IS: Moderate quality/diversity (< 3)")
        
        if results['lpips'] > 0.3:
            print("✓ LPIPS: Good perceptual diversity (> 0.3)")
        else:
            print("○ LPIPS: Low perceptual diversity (< 0.3)")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")

def example_data_analysis():
    """Example: Analyze dataset"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Dataset Analysis")
    print("="*50)
    
    data_path = "data/wellbore_images"
    
    print(f"Data path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Data path '{data_path}' does not exist.")
        print("Creating sample dataset for demonstration...")
        
        # Create sample dataset
        from examples.data_preparation import create_sample_dataset
        create_sample_dataset(data_path, num_samples=100, image_size=256)
    
    try:
        from data import DatasetAnalyzer
        
        # Create dataset
        dataset = WellboreImageDataset(
            data_path=data_path,
            image_size=256,
            augment=False
        )
        
        print(f"\nDataset created with {len(dataset)} images")
        
        # Analyze dataset
        print("\nAnalyzing dataset...")
        analyzer = DatasetAnalyzer(dataset)
        stats = analyzer.analyze()
        
        # Print analysis
        analyzer.print_analysis()
        
        # Create dataloader for batch analysis
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2
        )
        
        print(f"\nDataloader created:")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Total samples: {len(dataloader.dataset)}")
        
        # Sample a batch
        sample_batch = next(iter(dataloader))
        print(f"\nSample batch shape: {sample_batch.shape}")
        print(f"Sample batch range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
        
    except Exception as e:
        print(f"\nDataset analysis failed: {str(e)}")

def example_complete_pipeline():
    """Example: Complete pipeline from data preparation to evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Complete Pipeline")
    print("="*60)
    
    # Pipeline configuration
    base_dir = "pipeline_demo"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    print(f"Pipeline base directory: {base_dir}")
    
    try:
        # Step 1: Create sample data
        print("\n1. Creating sample dataset...")
        from examples.data_preparation import create_sample_dataset
        
        create_sample_dataset(
            output_dir=data_dir,
            num_samples=200,  # Small dataset for demo
            image_size=128    # Smaller size for faster training
        )
        print(f"   Sample dataset created: {data_dir}")
        
        # Step 2: Analyze data
        print("\n2. Analyzing dataset...")
        dataset = WellboreImageDataset(
            data_path=data_dir,
            image_size=128,
            augment=False
        )
        
        from data import DatasetAnalyzer
        analyzer = DatasetAnalyzer(dataset)
        stats = analyzer.analyze()
        print(f"   Dataset contains {stats['num_images']} images")
        print(f"   Mean pixel value: {stats['mean_pixel_value']:.3f}")
        
        # Step 3: Quick training (few epochs for demo)
        print("\n3. Training model (demo - few epochs)...")
        config = GANConfig(
            # Data settings
            data_path=data_dir,
            image_size=128,
            batch_size=8,  # Small batch for demo
            
            # Model settings
            latent_dim=256,  # Smaller latent dim
            num_mapping_layers=4,  # Fewer layers
            
            # Training settings
            num_epochs=5,  # Very few epochs for demo
            learning_rate=0.002,
            
            # Output settings
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            
            # Logging
            log_interval=10,
            save_interval=50,
            sample_interval=25
        )
        
        # Quick training
        train_model(config)
        print("   Training completed")
        
        # Step 4: Generate images
        print("\n4. Generating images...")
        model_path = os.path.join(checkpoint_dir, "stylegan2_generator_latest.pth")
        gen_output_dir = os.path.join(output_dir, "generated")
        
        generate_images(
            config=config,
            model_path=model_path,
            output_dir=gen_output_dir,
            num_images=16
        )
        print(f"   Images generated: {gen_output_dir}")
        
        # Step 5: Quick evaluation
        print("\n5. Evaluating model...")
        eval_output_dir = os.path.join(output_dir, "evaluation")
        
        results = evaluate_model(
            config=config,
            model_path=model_path,
            real_data_path=data_dir,
            output_dir=eval_output_dir,
            num_samples=100  # Small number for demo
        )
        
        print(f"   FID Score: {results['fid']:.4f}")
        print(f"   IS Score: {results['is_mean']:.4f}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"All outputs saved to: {base_dir}")
        print(f"  - Data: {data_dir}")
        print(f"  - Model: {checkpoint_dir}")
        print(f"  - Generated images: {gen_output_dir}")
        print(f"  - Evaluation: {eval_output_dir}")
        
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

def print_usage_guide():
    """Print usage guide"""
    print("\n" + "="*60)
    print("WELLBORE IMAGE GENERATION - USAGE GUIDE")
    print("="*60)
    
    print("\nThis script demonstrates various usage patterns for the")
    print("wellbore image generation system using StyleGAN2.")
    
    print("\nAvailable Examples:")
    print("  1. Training - Train a StyleGAN2 model from scratch")
    print("  2. Inference - Generate images using trained model")
    print("  3. Custom Generation - Advanced generation techniques")
    print("  4. Evaluation - Evaluate model quality with metrics")
    print("  5. Data Analysis - Analyze your dataset")
    print("  6. Complete Pipeline - End-to-end demonstration")
    
    print("\nQuick Start:")
    print("  1. Prepare your data:")
    print("     python data_preparation.py --action sample --output-dir data/wellbore_images")
    print("  ")
    print("  2. Train a model:")
    print("     python basic_usage.py --example training")
    print("  ")
    print("  3. Generate images:")
    print("     python basic_usage.py --example inference")
    print("  ")
    print("  4. Run complete pipeline:")
    print("     python basic_usage.py --example pipeline")
    
    print("\nFor more advanced usage, see the individual example functions.")
    print("\nRequirements:")
    print("  - PyTorch with CUDA support (recommended)")
    print("  - PIL, numpy, matplotlib")
    print("  - torchvision, scipy")
    print("  - Optional: opencv-python for video generation")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Basic usage examples for wellbore image generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--example',
        type=str,
        choices=['training', 'inference', 'custom', 'evaluation', 'analysis', 'pipeline', 'all'],
        default='all',
        help='Which example to run'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Print usage guide
    print_usage_guide()
    
    # Run examples
    if args.example == 'training' or args.example == 'all':
        example_training()
    
    if args.example == 'inference' or args.example == 'all':
        example_inference()
    
    if args.example == 'custom' or args.example == 'all':
        example_custom_generation()
    
    if args.example == 'evaluation' or args.example == 'all':
        example_evaluation()
    
    if args.example == 'analysis' or args.example == 'all':
        example_data_analysis()
    
    if args.example == 'pipeline':
        example_complete_pipeline()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nFor more information, check the documentation and source code.")
    print("Happy generating! 🎨")

if __name__ == '__main__':
    main()