"""Configuration settings for GAN-based wellbore image generation"""

import torch

class GANConfig:
    """Configuration class for StyleGAN2 training and inference"""
    
    # Model Architecture
    LATENT_DIM = 512
    IMAGE_SIZE = 256
    IMAGE_CHANNELS = 3
    
    # Generator Settings
    G_MAPPING_LAYERS = 8
    G_SYNTHESIS_LAYERS = 7
    STYLE_MIXING_PROB = 0.9
    
    # Discriminator Settings
    D_LAYERS = 7
    D_FEATURE_MAPS = 32
    
    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE_G = 0.002
    LEARNING_RATE_D = 0.002
    BETA1 = 0.0
    BETA2 = 0.99
    EPOCHS = 1000
    
    # Loss Weights
    ADVERSARIAL_LOSS_WEIGHT = 1.0
    R1_REGULARIZATION_WEIGHT = 10.0
    PATH_LENGTH_REGULARIZATION_WEIGHT = 2.0
    
    # Data Settings
    DATA_PATH = "data/wellbore_images"
    CHECKPOINT_PATH = "checkpoints"
    RESULTS_PATH = "results"
    
    # Device Settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # Wellbore Failure Types
    FAILURE_TYPES = [
        "breakouts",
        "drilling_induced_fractures", 
        "washouts",
        "casing_corrosion",
        "normal"
    ]
    
    # Logging
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1000
    SAMPLE_INTERVAL = 500