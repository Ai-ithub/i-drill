"""GAN Module for Wellbore Image Generation

This module contains the StyleGAN2 implementation for generating
synthetic wellbore images for predictive maintenance.
"""

from .generator import StyleGAN2Generator
from .discriminator import StyleGAN2Discriminator
from .trainer import GANTrainer

__all__ = ['StyleGAN2Generator', 'StyleGAN2Discriminator', 'GANTrainer']