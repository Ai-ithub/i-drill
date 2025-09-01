"""StyleGAN2 Generator Implementation for Wellbore Image Generation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

class EqualizedLinear(nn.Module):
    """Equalized learning rate linear layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 lr_multiplier: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_multiplier = lr_multiplier
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Calculate weight scale for equalized learning rate
        self.weight_scale = lr_multiplier / math.sqrt(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_scale
        return F.linear(x, weight, self.bias * self.lr_multiplier if self.bias is not None else None)

class EqualizedConv2d(nn.Module):
    """Equalized learning rate 2D convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True, lr_multiplier: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr_multiplier = lr_multiplier
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Calculate weight scale for equalized learning rate
        fan_in = in_channels * kernel_size * kernel_size
        self.weight_scale = lr_multiplier / math.sqrt(fan_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_scale
        return F.conv2d(x, weight, self.bias * self.lr_multiplier if self.bias is not None else None,
                       self.stride, self.padding)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.num_features = num_features
        self.style_dim = style_dim
        
        self.style_scale = EqualizedLinear(style_dim, num_features, bias=True)
        self.style_bias = EqualizedLinear(style_dim, num_features, bias=True)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Normalize input
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        normalized = (x - mean) / std
        
        # Apply style
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        
        return normalized * (1 + scale) + bias

class NoiseInjection(nn.Module):
    """Noise injection layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn(batch_size, 1, height, width, device=x.device)
        
        return x + self.weight * noise

class StyleBlock(nn.Module):
    """StyleGAN2 synthesis block"""
    
    def __init__(self, in_channels: int, out_channels: int, style_dim: int, 
                 upsample: bool = False, kernel_size: int = 3):
        super().__init__()
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # First convolution
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size, 
                                    padding=kernel_size//2, bias=False)
        self.noise1 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, style_dim)
        self.activation1 = nn.LeakyReLU(0.2)
        
        # Second convolution
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size,
                                    padding=kernel_size//2, bias=False)
        self.noise2 = NoiseInjection(out_channels)
        self.adain2 = AdaIN(out_channels, style_dim)
        self.activation2 = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, 
                noise1: Optional[torch.Tensor] = None, 
                noise2: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.upsample:
            x = self.upsample_layer(x)
        
        # First convolution block
        x = self.conv1(x)
        x = self.noise1(x, noise1)
        x = self.adain1(x, style)
        x = self.activation1(x)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.noise2(x, noise2)
        x = self.adain2(x, style)
        x = self.activation2(x)
        
        return x

class MappingNetwork(nn.Module):
    """StyleGAN2 mapping network (Z -> W)"""
    
    def __init__(self, latent_dim: int = 512, style_dim: int = 512, num_layers: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else style_dim
            layers.extend([
                EqualizedLinear(in_dim, style_dim),
                nn.LeakyReLU(0.2)
            ])
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize input
        z = F.normalize(z, dim=1)
        return self.mapping(z)

class SynthesisNetwork(nn.Module):
    """StyleGAN2 synthesis network (W -> Image)"""
    
    def __init__(self, style_dim: int = 512, num_layers: int = 7, 
                 base_channels: int = 512, max_channels: int = 512,
                 output_channels: int = 3, output_size: int = 256):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Calculate layer configurations
        self.layer_configs = self._calculate_layer_configs(base_channels, max_channels, num_layers)
        
        # Constant input (learned)
        self.const_input = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        
        # Style blocks
        self.blocks = nn.ModuleList()
        for i, (in_ch, out_ch, size) in enumerate(self.layer_configs):
            upsample = i > 0  # Don't upsample the first block
            self.blocks.append(StyleBlock(in_ch, out_ch, style_dim, upsample))
        
        # Output layer (to RGB)
        final_channels = self.layer_configs[-1][1]
        self.to_rgb = EqualizedConv2d(final_channels, output_channels, 1)
    
    def _calculate_layer_configs(self, base_channels: int, max_channels: int, 
                                num_layers: int) -> List[Tuple[int, int, int]]:
        """Calculate channel configurations for each layer"""
        configs = []
        current_size = 4
        
        for i in range(num_layers):
            if i == 0:
                in_ch = base_channels
                out_ch = min(base_channels, max_channels)
            else:
                in_ch = configs[-1][1]
                # Reduce channels as we go up in resolution
                out_ch = max(base_channels // (2 ** (i-1)), base_channels // 8)
                out_ch = min(out_ch, max_channels)
                current_size *= 2
            
            configs.append((in_ch, out_ch, current_size))
        
        return configs
    
    def forward(self, styles: torch.Tensor, noise_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        batch_size = styles.shape[0]
        
        # Start with constant input
        x = self.const_input.repeat(batch_size, 1, 1, 1)
        
        # Apply style blocks
        for i, block in enumerate(self.blocks):
            # Get noise for this layer
            noise1 = noise_inputs[i*2] if noise_inputs else None
            noise2 = noise_inputs[i*2+1] if noise_inputs else None
            
            x = block(x, styles, noise1, noise2)
        
        # Convert to RGB
        x = self.to_rgb(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        
        return x

class StyleGAN2Generator(nn.Module):
    """Complete StyleGAN2 Generator"""
    
    def __init__(self, latent_dim: int = 512, style_dim: int = 512,
                 num_mapping_layers: int = 8, num_synthesis_layers: int = 7,
                 output_channels: int = 3, output_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.output_size = output_size
        
        self.mapping = MappingNetwork(latent_dim, style_dim, num_mapping_layers)
        self.synthesis = SynthesisNetwork(style_dim, num_synthesis_layers, 
                                        output_channels=output_channels,
                                        output_size=output_size)
    
    def forward(self, z: torch.Tensor, truncation_psi: float = 1.0,
                noise_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # Map latent code to style space
        w = self.mapping(z)
        
        # Apply truncation trick if specified
        if truncation_psi < 1.0:
            w_avg = w.mean(dim=0, keepdim=True)
            w = w_avg + truncation_psi * (w - w_avg)
        
        # Generate image
        return self.synthesis(w, noise_inputs)
    
    def generate_noise(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Generate noise inputs for synthesis network"""
        noise_inputs = []
        current_size = 4
        
        for i in range(len(self.synthesis.blocks)):
            # Two noise inputs per block
            noise1 = torch.randn(batch_size, 1, current_size, current_size, device=device)
            noise2 = torch.randn(batch_size, 1, current_size, current_size, device=device)
            noise_inputs.extend([noise1, noise2])
            
            if i > 0:  # Size doubles after first block
                current_size *= 2
        
        return noise_inputs