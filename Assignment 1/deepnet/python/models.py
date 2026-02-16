"""Model building utilities"""

import yaml
from pathlib import Path

import sys
import os
build_path = os.path.join(os.path.dirname(__file__), '../../build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

# Add utils to path
utils_path = os.path.join(os.path.dirname(__file__), '../../utils')
if os.path.exists(utils_path):
    sys.path.insert(0, utils_path)

import deepnet_backend as backend
from .module import (Module, Sequential, Conv2DWrapper, LinearWrapper, ReLUWrapper,
                     MaxPool2DWrapper, BatchNorm2DWrapper, BatchNorm1DWrapper,
                     DropoutWrapper, FlattenWrapper, LeakyReLUWrapper,
                     TanhWrapper, SigmoidWrapper, AvgPool2DWrapper,
                     GlobalAvgPool2DWrapper, ResidualBlockWrapper)

try:
    from metrics import count_parameters, estimate_macs_flops
except ImportError:
    # Fallback if utils not in path
    def count_parameters(model):
        total = 0
        params = model.parameters()
        for param in params:
            param_count = 1
            for dim in param.shape:
                param_count *= dim
            total += param_count
        return total
    
    def estimate_macs_flops(model, input_shape):
        return {'macs': 0, 'flops': 0}


def build_model_from_config(config_or_path, num_classes):
    """Build model from YAML configuration file or dictionary"""
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        with open(config_or_path, 'r') as f:
            config = yaml.safe_load(f)
    
    architecture = config['model']['architecture']
    layers = []
    
    for layer_config in architecture:
        layer_type = layer_config['type']
        
        if layer_type == 'Conv2D':
            # Update out_features if it's the last layer
            layers.append(Conv2DWrapper(
                in_channels=layer_config['in_channels'],
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config.get('stride', 1),
                padding=layer_config.get('padding', 0),
                bias=layer_config.get('bias', True)
            ))
        
        elif layer_type == 'Linear':
            out_features = layer_config['out_features']
            # Support "num_classes" string placeholder or legacy integer 10
            if out_features == "num_classes" or out_features == 10:
                out_features = num_classes
            
            layers.append(LinearWrapper(
                in_features=layer_config['in_features'],
                out_features=out_features,
                bias=layer_config.get('bias', True)
            ))
        
        elif layer_type == 'ReLU':
            layers.append(ReLUWrapper())
        
        elif layer_type == 'MaxPool2D':
            layers.append(MaxPool2DWrapper(
                kernel_size=layer_config['kernel_size'],
                stride=layer_config.get('stride')
            ))
        
        elif layer_type == 'BatchNorm2D':
            layers.append(BatchNorm2DWrapper(
                num_features=layer_config['num_features'],
                eps=layer_config.get('eps', 1e-5),
                momentum=layer_config.get('momentum', 0.1)
            ))
        
        elif layer_type == 'BatchNorm1D':
            layers.append(BatchNorm1DWrapper(
                num_features=layer_config['num_features'],
                eps=layer_config.get('eps', 1e-5),
                momentum=layer_config.get('momentum', 0.1)
            ))
        
        elif layer_type == 'Dropout':
            layers.append(DropoutWrapper(
                p=layer_config.get('p', 0.5)
            ))
        
        elif layer_type == 'Flatten':
            layers.append(FlattenWrapper())
        
        elif layer_type == 'LeakyReLU':
            layers.append(LeakyReLUWrapper(
                negative_slope=layer_config.get('negative_slope', 0.01)
            ))
        
        elif layer_type == 'Tanh':
            layers.append(TanhWrapper())
        
        elif layer_type == 'Sigmoid':
            layers.append(SigmoidWrapper())
        
        elif layer_type == 'AvgPool2D':
            layers.append(AvgPool2DWrapper(
                kernel_size=layer_config['kernel_size'],
                stride=layer_config.get('stride')
            ))
        
        elif layer_type == 'GlobalAveragePooling2D':
            layers.append(GlobalAvgPool2DWrapper())
        
        elif layer_type == 'ResidualBlock':
            layers.append(ResidualBlockWrapper(
                in_channels=layer_config['in_channels'],
                out_channels=layer_config['out_channels'],
                stride=layer_config.get('stride', 1)
            ))
        
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    return Sequential(*layers), config


def calculate_model_stats(model, input_shape):
    """Calculate number of parameters, MACs, and FLOPs"""
    total_params = count_parameters(model)
    metrics = estimate_macs_flops(model, input_shape)
    
    return {
        'parameters': total_params,
        'macs': metrics['macs'],
        'flops': metrics['flops']
    }


def save_checkpoint(model, path, optimizer=None, epoch=None, loss=None, config=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    # Optimizer state saving could be added here if backend supports it
    if optimizer and hasattr(optimizer, 'state_dict'):
        checkpoint['optimizer_state'] = optimizer.state_dict()
    
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(model, path, optimizer=None):
    """Load model checkpoint"""
    import pickle
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model.load_state_dict(checkpoint['model_state'])
    
    # Optimizer state loading could be added here if backend supports it
    if optimizer and 'optimizer_state' in checkpoint and hasattr(optimizer, 'load_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    return checkpoint.get('epoch'), checkpoint.get('loss'), checkpoint.get('config')

