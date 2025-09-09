#!/usr/bin/env python3
"""
Main Script for RUL Prediction

This script provides a complete pipeline for training and evaluating
deep learning models for remaining useful life prediction.
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Any

# Import RUL prediction modules
from data_loader import RULDataLoader
from models import create_model
from trainer import RULTrainer
from evaluator import RULEvaluator

def setup_logging(log_level: str = 'INFO'):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rul_prediction.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            'data_path': './data/task31_processed',
            'sequence_length': 50,
            'test_size': 0.2,
            'val_size': 0.1,
            'scaler_type': 'standard'
        },
        'model': {
            'type': 'lstm',  # 'lstm', 'transformer', 'cnn_lstm'
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler_type': 'cosine',
            'patience': 10,
            'batch_size': 32
        },
        'evaluation': {
            'detailed_analysis': True,
            'save_predictions': True
        },
        'paths': {
            'checkpoints': './checkpoints',
            'results': './results'
        }
    }

def create_config_file(config_path: str):
    """
    Create a default configuration file
    
    Args:
        config_path: Path where to save the config file
    """
    config = get_default_config()
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {config_path}")
    print("Please modify the configuration as needed and run the script again.")

def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train RUL prediction model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting RUL model training...")
    
    # Initialize data loader
    data_loader = RULDataLoader(
        data_path=config['data']['data_path'],
        sequence_length=config['data']['sequence_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        scaler_type=config['data']['scaler_type']
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = data_loader.prepare_data()
    input_dim = data_loader.get_feature_dim()
    
    logger.info(f"Data prepared - Input dimension: {input_dim}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(
        model_type=config['model']['type'],
        input_dim=input_dim,
        **{k: v for k, v in config['model'].items() if k != 'type'}
    )
    
    logger.info(f"Created {config['model']['type'].upper()} model")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = RULTrainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler_type'],
        patience=config['training']['patience'],
        save_dir=config['paths']['checkpoints']
    )
    
    # Train model
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        save_best=True,
        plot_progress=True
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'data_loader': data_loader,
        'test_loader': test_loader,
        'training_history': training_history
    }

def evaluate_model(model: torch.nn.Module, test_loader, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = RULEvaluator(
        model=model,
        save_dir=config['paths']['results']
    )
    
    # Evaluate model
    evaluation_results = evaluator.evaluate(
        test_loader=test_loader,
        detailed_analysis=config['evaluation']['detailed_analysis']
    )
    
    return evaluation_results

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='RUL Prediction with Deep Learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'],
                       default='both', help='Mode: train, evaluate, or both')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained model (for evaluation mode)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create config file if requested
    if args.create_config:
        create_config_file(args.config)
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    Path(config['paths']['checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results']).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting RUL prediction pipeline in '{args.mode}' mode")
    logger.info(f"Configuration: {args.config}")
    
    try:
        if args.mode in ['train', 'both']:
            # Training mode
            training_results = train_model(config)
            
            if args.mode == 'both':
                # Evaluate the trained model
                evaluation_results = evaluate_model(
                    model=training_results['model'],
                    test_loader=training_results['test_loader'],
                    config=config
                )
                
                logger.info("Training and evaluation completed successfully!")
                
        elif args.mode == 'evaluate':
            # Evaluation mode with pre-trained model
            if args.model_path is None:
                raise ValueError("Model path must be specified for evaluation mode")
            
            # Load data
            data_loader = RULDataLoader(
                data_path=config['data']['data_path'],
                sequence_length=config['data']['sequence_length'],
                test_size=config['data']['test_size'],
                val_size=config['data']['val_size'],
                scaler_type=config['data']['scaler_type']
            )
            
            _, _, test_loader = data_loader.prepare_data()
            input_dim = data_loader.get_feature_dim()
            
            # Load model
            model = create_model(
                model_type=config['model']['type'],
                input_dim=input_dim,
                **{k: v for k, v in config['model'].items() if k != 'type'}
            )
            
            # Load checkpoint
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            evaluation_results = evaluate_model(model, test_loader, config)
            
            logger.info("Evaluation completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
    
    logger.info("RUL prediction pipeline completed!")

if __name__ == '__main__':
    main()