#!/usr/bin/env python3
"""Model utilities for wellbore image generation system"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from collections import OrderedDict

def weights_init(m: nn.Module, init_type: str = 'xavier_normal', gain: float = 0.02):
    """Initialize network weights
    
    Args:
        m: Module to initialize
        init_type: Type of initialization ('xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', 'normal', 'orthogonal')
        gain: Gain factor for initialization
    """
    classname = m.__class__.__name__
    
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data, gain=gain)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError(f'Initialization method {init_type} is not implemented')
        
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)
    
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)

def gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, 
                    fake_samples: torch.Tensor, device: torch.device,
                    lambda_gp: float = 10.0) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP
    
    Args:
        discriminator: Discriminator network
        real_samples: Real image samples
        fake_samples: Generated image samples
        device: Device to perform computation on
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients of discriminator output w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Calculate gradient penalty
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def calculate_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calculate model size and memory requirements
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        'param_count': param_sum,
        'buffer_count': buffer_sum,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': all_size,
        'total_size_gb': all_size / 1024
    }

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters in a model
    
    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters in a model
    
    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True

def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze specific layers in a model
    
    Args:
        model: Model containing layers to freeze
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False

def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Unfreeze specific layers in a model
    
    Args:
        model: Model containing layers to unfreeze
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

def apply_spectral_norm(model: nn.Module, layer_types: Optional[List[type]] = None) -> nn.Module:
    """Apply spectral normalization to specified layer types
    
    Args:
        model: Model to apply spectral norm to
        layer_types: List of layer types to apply spectral norm to
        
    Returns:
        Model with spectral normalization applied
    """
    if layer_types is None:
        layer_types = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]
    
    for name, module in model.named_modules():
        if type(module) in layer_types:
            spectral_norm(module)
    
    return model

def remove_spectral_norm(model: nn.Module) -> nn.Module:
    """Remove spectral normalization from all layers
    
    Args:
        model: Model to remove spectral norm from
        
    Returns:
        Model with spectral normalization removed
    """
    for name, module in model.named_modules():
        try:
            nn.utils.remove_spectral_norm(module)
        except ValueError:
            # Module doesn't have spectral norm
            pass
    
    return model

def get_model_summary(model: nn.Module, input_size: Tuple[int, ...], 
                     device: str = 'cuda') -> str:
    """Get a detailed summary of the model architecture
    
    Args:
        model: Model to summarize
        input_size: Input tensor size (without batch dimension)
        device: Device to run the model on
        
    Returns:
        Model summary string
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Create summary dict
    summary = OrderedDict()
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    model.eval()
    x = torch.rand(1, *input_size).to(device)
    
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Create summary string
    summary_str = ""
    summary_str += "-" * 80 + "\n"
    summary_str += f"{'Layer (type)':>25} {'Output Shape':>25} {'Param #':>15}\n"
    summary_str += "=" * 80 + "\n"
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        # Input shape
        line_new = f"{layer:>25} {str(summary[layer]['output_shape']):>25} {summary[layer]['nb_params']:>15,}\n"
        total_params += summary[layer]["nb_params"]
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        
        summary_str += line_new
    
    # Calculate model size
    model_size = calculate_model_size(model)
    
    summary_str += "=" * 80 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
    summary_str += f"Model size: {model_size['total_size_mb']:.2f} MB\n"
    summary_str += "-" * 80 + "\n"
    
    return summary_str

def calculate_receptive_field(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
    """Calculate the receptive field of a model
    
    Args:
        model: Model to analyze
        input_size: Input tensor size
        
    Returns:
        Dictionary with receptive field information
    """
    def conv_output_size(input_size, kernel_size, stride, padding, dilation=1):
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    def conv_receptive_field(input_rf, kernel_size, stride, padding, dilation=1):
        return input_rf + (kernel_size - 1) * dilation
    
    # This is a simplified calculation - for more complex architectures,
    # you might need a more sophisticated approach
    receptive_field = 1
    current_size = input_size[-1]  # Assuming square images
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            dilation = getattr(module, 'dilation', 1)
            if isinstance(dilation, tuple):
                dilation = dilation[0]
            
            receptive_field = conv_receptive_field(receptive_field, kernel_size, stride, padding, dilation)
            current_size = conv_output_size(current_size, kernel_size, stride, padding, dilation)
        
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            
            receptive_field = conv_receptive_field(receptive_field, kernel_size, stride, padding)
            current_size = conv_output_size(current_size, kernel_size, stride, padding)
    
    return {
        'receptive_field': receptive_field,
        'final_feature_size': current_size,
        'receptive_field_ratio': receptive_field / input_size[-1]
    }

def load_pretrained_weights(model: nn.Module, pretrained_path: str, 
                          strict: bool = True, prefix: str = '') -> None:
    """Load pretrained weights into a model
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights
        strict: Whether to strictly enforce that the keys match
        prefix: Prefix to add to state dict keys
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Add prefix if specified
    if prefix:
        state_dict = {f'{prefix}.{k}': v for k, v in state_dict.items()}
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

def exponential_moving_average(model: nn.Module, ema_model: nn.Module, 
                             decay: float = 0.999) -> None:
    """Update exponential moving average of model parameters
    
    Args:
        model: Current model
        ema_model: EMA model to update
        decay: EMA decay factor
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def copy_model_weights(source_model: nn.Module, target_model: nn.Module) -> None:
    """Copy weights from source model to target model
    
    Args:
        source_model: Model to copy weights from
        target_model: Model to copy weights to
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)

def get_layer_names(model: nn.Module, layer_type: Optional[type] = None) -> List[str]:
    """Get names of all layers of a specific type
    
    Args:
        model: Model to analyze
        layer_type: Type of layer to find (None for all layers)
        
    Returns:
        List of layer names
    """
    layer_names = []
    
    for name, module in model.named_modules():
        if layer_type is None or isinstance(module, layer_type):
            layer_names.append(name)
    
    return layer_names

def replace_activation(model: nn.Module, old_activation: type, 
                     new_activation: nn.Module) -> nn.Module:
    """Replace all instances of an activation function with a new one
    
    Args:
        model: Model to modify
        old_activation: Type of activation to replace
        new_activation: New activation function instance
        
    Returns:
        Modified model
    """
    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation)
        else:
            replace_activation(module, old_activation, new_activation)
    
    return model