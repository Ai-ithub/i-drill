"""StyleGAN2 Discriminator Implementation for Wellbore Image Generation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

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

class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for improved diversity"""
    
    def __init__(self, group_size: int = 4, num_new_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Reshape for group processing
        group_size = min(batch_size, self.group_size)
        y = x.view(group_size, -1, self.num_new_features, 
                  channels // self.num_new_features, height, width)
        
        # Calculate standard deviation across the group
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        
        # Average over feature maps and spatial dimensions
        y = y.mean(dim=[2, 3, 4], keepdim=True)
        y = y.mean(dim=2)
        
        # Replicate across batch and spatial dimensions
        y = y.repeat(group_size, 1, height, width)
        
        # Handle remaining batch items
        if batch_size > group_size:
            remaining = batch_size - group_size
            y_remaining = y[:1].repeat(remaining, 1, 1, 1)
            y = torch.cat([y, y_remaining], dim=0)
        
        return torch.cat([x, y], dim=1)

class DiscriminatorBlock(nn.Module):
    """StyleGAN2 discriminator block"""
    
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        
        # First convolution
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.activation1 = nn.LeakyReLU(0.2)
        
        # Second convolution
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.activation2 = nn.LeakyReLU(0.2)
        
        # Downsampling
        if downsample:
            self.downsample_layer = nn.AvgPool2d(2)
        
        # Skip connection
        if in_channels != out_channels or downsample:
            self.skip = EqualizedConv2d(in_channels, out_channels, 1)
            if downsample:
                self.skip_downsample = nn.AvgPool2d(2)
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.activation2(out)
        
        if self.downsample:
            out = self.downsample_layer(out)
        
        # Skip connection
        if self.skip is not None:
            skip = self.skip(x)
            if hasattr(self, 'skip_downsample'):
                skip = self.skip_downsample(skip)
        else:
            skip = x
        
        return (out + skip) / math.sqrt(2)

class FromRGB(nn.Module):
    """Convert RGB input to feature maps"""
    
    def __init__(self, out_channels: int, input_channels: int = 3):
        super().__init__()
        self.conv = EqualizedConv2d(input_channels, out_channels, 1)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))

class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator Network"""
    
    def __init__(self, input_channels: int = 3, input_size: int = 256, 
                 base_channels: int = 32, max_channels: int = 512,
                 num_layers: int = 7):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_layers = num_layers
        
        # Calculate layer configurations
        self.layer_configs = self._calculate_layer_configs(base_channels, max_channels, num_layers)
        
        # From RGB layer
        self.from_rgb = FromRGB(self.layer_configs[0][0], input_channels)
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.layer_configs):
            downsample = i < len(self.layer_configs) - 1  # Don't downsample the last block
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch, downsample))
        
        # Minibatch standard deviation
        final_channels = self.layer_configs[-1][1]
        self.minibatch_stddev = MinibatchStdDev()
        
        # Final layers
        self.final_conv = EqualizedConv2d(final_channels + 1, final_channels, 3, padding=1)
        self.final_activation = nn.LeakyReLU(0.2)
        
        # Calculate final spatial size (should be 4x4 after all downsampling)
        final_spatial_size = 4
        self.final_linear = EqualizedLinear(final_channels * final_spatial_size * final_spatial_size, 
                                          final_channels)
        self.output_linear = EqualizedLinear(final_channels, 1)
    
    def _calculate_layer_configs(self, base_channels: int, max_channels: int, 
                                num_layers: int) -> list:
        """Calculate channel configurations for each layer"""
        configs = []
        
        for i in range(num_layers):
            if i == 0:
                in_ch = base_channels
                out_ch = min(base_channels * 2, max_channels)
            else:
                in_ch = configs[-1][1]
                # Increase channels as we go down in resolution
                out_ch = min(in_ch * 2, max_channels)
            
            configs.append((in_ch, out_ch))
        
        return configs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from RGB
        x = self.from_rgb(x)
        
        # Apply discriminator blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply minibatch standard deviation
        x = self.minibatch_stddev(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        # Flatten and apply final linear layers
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        x = self.final_activation(x)
        x = self.output_linear(x)
        
        return x
    
    def get_features(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Extract features from a specific layer for analysis"""
        # Convert from RGB
        x = self.from_rgb(x)
        
        # Apply discriminator blocks up to specified layer
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer_idx:
                break
        
        return x

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for improved training stability"""
    
    def __init__(self, input_channels: int = 3, input_size: int = 256,
                 num_discriminators: int = 2):
        super().__init__()
        self.num_discriminators = num_discriminators
        
        # Create multiple discriminators at different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            scale_factor = 2 ** i
            scaled_size = input_size // scale_factor
            self.discriminators.append(
                StyleGAN2Discriminator(input_channels, scaled_size)
            )
        
        # Downsampling layers
        self.downsample = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass through all discriminators"""
        results = []
        current_x = x
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                current_x = self.downsample(current_x)
            
            result = discriminator(current_x)
            results.append(result)
        
        return results