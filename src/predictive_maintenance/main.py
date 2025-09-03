#!/usr/bin/env python3
"""Main entry point for Wellbore Image Generation using StyleGAN2"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import GANConfig
from train import train_gan
from inference import generate_images
from evaluation import evaluate_model
from utils import setup_logging, ensure_dir

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Wellbore Image Generation using StyleGAN2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'inference', 'evaluate'], 
        required=True,
        help='Operation mode: train, inference, or evaluate'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/training_config_example.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for generated images or results'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to model checkpoint for inference or evaluation'
    )
    
    parser.add_argument(
        '--num-images', 
        type=int, 
        default=100,
        help='Number of images to generate (inference mode)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for computation'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup computation device"""
    import torch
    
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    logging.info(f"Using device: {device}")
    return device

def setup_random_seed(seed):
    """Setup random seed for reproducibility"""
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logging.info(f"Random seed set to: {seed}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logging.info("Starting Wellbore Image Generation System")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Config: {args.config}")
    
    # Setup device and random seed
    device = setup_device(args.device)
    setup_random_seed(args.seed)
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        config = GANConfig.from_yaml(args.config)
        
        # Override config with command line arguments
        config.DEVICE = device
        if args.batch_size is not None:
            config.BATCH_SIZE = args.batch_size
        
        # Execute based on mode
        if args.mode == 'train':
            logging.info("Starting training...")
            train_gan(config, args.output_dir)
            
        elif args.mode == 'inference':
            logging.info("Starting inference...")
            if args.checkpoint is None:
                raise ValueError("Checkpoint path required for inference mode")
            
            generate_images(
                config=config,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                num_images=args.num_images
            )
            
        elif args.mode == 'evaluate':
            logging.info("Starting evaluation...")
            if args.checkpoint is None:
                raise ValueError("Checkpoint path required for evaluation mode")
            
            evaluate_model(
                config=config,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir
            )
        
        logging.info(f"Operation completed successfully. Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()