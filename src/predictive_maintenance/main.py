#!/usr/bin/env python3
"""
Main entry point for wellbore image generation system

This script provides a command-line interface for training and inference
of GAN models for wellbore image generation.

Usage:
    # Training
    python main.py train --config config/training_config.yaml
    
    # Inference
    python main.py generate --model checkpoints/best_model.pth --num_images 100
    
    # Evaluation
    python main.py evaluate --model checkpoints/best_model.pth --data_dir data/test
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import torch
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.predictive_maintenance.config import TrainingConfig, InferenceConfig, ConfigManager
from src.predictive_maintenance.train import GANTrainer
from src.predictive_maintenance.inference import GANInference
from src.predictive_maintenance.evaluation import ModelEvaluator
from src.predictive_maintenance.utils.training_utils import setup_logging, set_seed
from src.predictive_maintenance.utils.file_utils import ensure_dir

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Wellbore Image Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --config config/training_config.yaml
  
  # Resume training from checkpoint
  python main.py train --config config/training_config.yaml --resume checkpoints/latest.pth
  
  # Generate images
  python main.py generate --model checkpoints/best_model.pth --num_images 50 --output generated/
  
  # Evaluate model
  python main.py evaluate --model checkpoints/best_model.pth --data_dir data/test --output evaluation/
  
  # Create default config
  python main.py create-config --output config/default_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train GAN model')
    train_parser.add_argument('--config', '-c', type=str, required=True,
                             help='Path to training configuration file')
    train_parser.add_argument('--resume', '-r', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--gpu', '-g', type=int, default=None,
                             help='GPU device ID to use (default: auto-select)')
    train_parser.add_argument('--seed', '-s', type=int, default=42,
                             help='Random seed for reproducibility')
    train_parser.add_argument('--debug', action='store_true',
                             help='Enable debug mode')
    
    # Generation command
    generate_parser = subparsers.add_parser('generate', help='Generate images')
    generate_parser.add_argument('--model', '-m', type=str, required=True,
                                help='Path to trained model checkpoint')
    generate_parser.add_argument('--config', '-c', type=str, default=None,
                                help='Path to inference configuration file')
    generate_parser.add_argument('--num_images', '-n', type=int, default=10,
                                help='Number of images to generate')
    generate_parser.add_argument('--output', '-o', type=str, default='generated/',
                                help='Output directory for generated images')
    generate_parser.add_argument('--batch_size', '-b', type=int, default=8,
                                help='Batch size for generation')
    generate_parser.add_argument('--seed', '-s', type=int, default=None,
                                help='Random seed for generation')
    generate_parser.add_argument('--format', '-f', type=str, default='png',
                                choices=['png', 'jpg', 'jpeg'],
                                help='Output image format')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', '-m', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--data_dir', '-d', type=str, required=True,
                            help='Path to real images directory for comparison')
    eval_parser.add_argument('--config', '-c', type=str, default=None,
                            help='Path to evaluation configuration file')
    eval_parser.add_argument('--output', '-o', type=str, default='evaluation/',
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--num_samples', '-n', type=int, default=1000,
                            help='Number of samples to generate for evaluation')
    eval_parser.add_argument('--batch_size', '-b', type=int, default=32,
                            help='Batch size for evaluation')
    
    # Config creation command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration file')
    config_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output path for configuration file')
    config_parser.add_argument('--type', '-t', type=str, default='training',
                              choices=['training', 'inference'],
                              help='Type of configuration to create')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--model', '-m', type=str, default=None,
                            help='Path to model checkpoint to analyze')
    
    return parser

def train_command(args):
    """Execute training command
    
    Args:
        args: Parsed command line arguments
    """
    print("🚀 Starting GAN training...")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_training_config(args.config)
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else config.logging.level
    logger = setup_logging(config.logging.log_dir, log_level)
    
    # Override GPU setting if specified
    if args.gpu is not None:
        config.training.device_id = args.gpu
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed training from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    try:
        # Start training
        trainer.train(start_epoch=start_epoch)
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False, 
                              filename='interrupted_checkpoint.pth')
        print("💾 Checkpoint saved as 'interrupted_checkpoint.pth'")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        sys.exit(1)

def generate_command(args):
    """Execute generation command
    
    Args:
        args: Parsed command line arguments
    """
    print(f"🎨 Generating {args.num_images} images...")
    
    # Load configuration
    if args.config:
        config_manager = ConfigManager()
        config = config_manager.load_inference_config(args.config)
    else:
        config = InferenceConfig()
    
    # Override settings from command line
    config.generation.num_images = args.num_images
    config.generation.batch_size = args.batch_size
    config.generation.output_dir = args.output
    config.generation.image_format = args.format
    
    if args.seed is not None:
        config.generation.seed = args.seed
        set_seed(args.seed)
    
    # Create inference engine
    inference = GANInference(config)
    
    # Load model
    if not inference.load_model(args.model):
        print(f"❌ Failed to load model: {args.model}")
        sys.exit(1)
    
    try:
        # Generate images
        output_paths = inference.generate_images()
        print(f"✅ Generated {len(output_paths)} images successfully!")
        print(f"📁 Images saved to: {args.output}")
        
        # Show some sample paths
        if output_paths:
            print("\n📸 Sample generated images:")
            for i, path in enumerate(output_paths[:5]):
                print(f"  {i+1}. {path}")
            if len(output_paths) > 5:
                print(f"  ... and {len(output_paths) - 5} more")
                
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)

def evaluate_command(args):
    """Execute evaluation command
    
    Args:
        args: Parsed command line arguments
    """
    print(f"📊 Evaluating model performance...")
    
    # Load configuration
    if args.config:
        config_manager = ConfigManager()
        config = config_manager.load_inference_config(args.config)
    else:
        config = InferenceConfig()
    
    # Override settings
    config.generation.num_images = args.num_samples
    config.generation.batch_size = args.batch_size
    config.generation.output_dir = os.path.join(args.output, 'generated')
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Load model
    if not evaluator.load_model(args.model):
        print(f"❌ Failed to load model: {args.model}")
        sys.exit(1)
    
    try:
        # Run evaluation
        results = evaluator.evaluate(args.data_dir, args.output)
        
        print("✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        # Display key metrics
        print("\n📈 Key Metrics:")
        if 'fid' in results:
            print(f"  FID Score: {results['fid']:.4f}")
        if 'is_mean' in results:
            print(f"  Inception Score: {results['is_mean']:.4f} ± {results.get('is_std', 0):.4f}")
        if 'ssim' in results:
            print(f"  SSIM Score: {results['ssim']:.4f}")
        if 'lpips' in results:
            print(f"  LPIPS Score: {results['lpips']:.4f}")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

def create_config_command(args):
    """Execute config creation command
    
    Args:
        args: Parsed command line arguments
    """
    print(f"📝 Creating {args.type} configuration file...")
    
    try:
        # Ensure output directory exists
        output_path = Path(args.output)
        ensure_dir(output_path.parent)
        
        # Create configuration
        config_manager = ConfigManager()
        
        if args.type == 'training':
            config = TrainingConfig()
            config_manager.save_training_config(config, args.output)
        else:
            config = InferenceConfig()
            config_manager.save_inference_config(config, args.output)
        
        print(f"✅ Configuration file created: {args.output}")
        print("\n📋 You can now edit this file to customize your settings.")
        
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        sys.exit(1)

def info_command(args):
    """Execute info command
    
    Args:
        args: Parsed command line arguments
    """
    print("ℹ️ System Information")
    print("=" * 50)
    
    # Python and PyTorch info
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Project info
    print(f"\nProject Root: {project_root}")
    
    # Model info
    if args.model and os.path.exists(args.model):
        print(f"\n📦 Model Information")
        print("-" * 30)
        
        try:
            checkpoint = torch.load(args.model, map_location='cpu')
            
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            if 'iteration' in checkpoint:
                print(f"Iteration: {checkpoint['iteration']}")
            if 'best_score' in checkpoint:
                print(f"Best Score: {checkpoint['best_score']}")
            if 'config' in checkpoint:
                print(f"Model Config: Available")
            
            # Model size
            file_size = os.path.getsize(args.model)
            print(f"File Size: {file_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load model info: {e}")

def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'generate':
            generate_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'create-config':
            create_config_command(args)
        elif args.command == 'info':
            info_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()