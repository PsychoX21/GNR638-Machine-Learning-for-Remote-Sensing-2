"""PyTorch-like Module abstraction"""

import sys
import os

# Add build directory to path
build_path = os.path.join(os.path.dirname(__file__), '../../build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    import deepnet_backend as backend
except ImportError:
    print("ERROR: Could not import deepnet_backend. Please run 'make build install' first.")
    sys.exit(1)

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._modules = {}
        self._parameters = []
        self._training = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        """Make module callable"""
        return self.forward(*args, **kwargs)
    
    def add_module(self, name, module):
        """Add a child module"""
        self._modules[name] = module
        setattr(self, name, module)
    
    def parameters(self):
        """Get all parameters recursively"""
        params = []
        # Get own parameters
        params.extend(self._parameters)
        # Get parameters from child modules
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
            elif hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def train(self):
        """Set module to training mode"""
        self._training = True
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train()
            elif hasattr(module, 'train'):
                module.train()
    
    def eval(self):
        """Set module to evaluation mode"""
        self._training = False
        for module in self._modules.values():
            if isinstance(module, Module):
                module.eval()
            elif hasattr(module, 'eval'):
                module.eval()
    
    
    def cuda(self):
        """Move all parameters to CUDA"""
        for param in self.parameters():
            if hasattr(param, 'cuda'):
                param.cuda()
        
        # Recursively call cuda() on children that are Modules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.cuda()
            elif hasattr(module, 'cuda'):
                module.cuda()
        return self

    def cpu(self):
        """Move all parameters to CPU"""
        for param in self.parameters():
            if hasattr(param, 'cpu'):
                param.cpu()
        
        # Recursively call cpu() on children that are Modules
        for module in self._modules.values():
            if isinstance(module, Module):
                module.cpu()
            elif hasattr(module, 'cpu'):
                module.cpu()
        return self

    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters():
            param.zero_grad()
    
    def state_dict(self):
        """Get state dictionary for saving"""
        state = {}
        for i, param in enumerate(self.parameters()):
            state[f'param_{i}'] = {
                'data': param.data,
                'shape': param.shape,
                'requires_grad': param.requires_grad
            }
        return state
    
    def load_state_dict(self, state):
        """Load state dictionary"""
        params = self.parameters()
        for i, param in enumerate(params):
            if f'param_{i}' in state:
                saved_item = state[f'param_{i}']
                saved_data = saved_item['data']
                
                # Validation
                if isinstance(saved_data, list):
                    # Check size matches shape
                    expected_count = 1
                    for dim in param.shape:
                        expected_count *= dim
                        
                    if len(saved_data) != expected_count:
                        print(f"WARNING: Param {i} size mismatch! Shape {param.shape} needs {expected_count}, got {len(saved_data)}")
                    
                # Load parameter data
                try:
                    # Create target tensor from saved data
                    target = backend.Tensor.from_data(saved_data, param.shape)
                    
                    # Use in-place copy (requires updated backend with copy_)
                    if hasattr(param, 'copy_'):
                        param.copy_(target)
                    else:
                        # Fallback for older backends
                        param.data = saved_data
                        
                except Exception as e:
                    print(f"Warning: Failed to update parameter {i}: {e}")
                    param.data = saved_data



class Sequential(Module):
    """Sequential container for layers"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            self.add_module(f'layer_{i}', layer)
    
    def forward(self, x):
        """Forward pass through all layers"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                x = layer(x)
            elif hasattr(layer, 'forward'):
                x = layer.forward(x)
            else:
                raise ValueError(f"Invalid layer type: {type(layer)}")
        return x
    
    def backward(self, grad):
        """Backward pass through all layers in reverse"""
        for layer in reversed(self.layers):
            if isinstance(layer, Module):
                grad = layer.backward(grad)
            elif hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        """Get all parameters from all layers"""
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
            elif hasattr(layer, 'parameters'):
                layer_params = layer.parameters()
                if layer_params:
                    params.extend(layer_params)
        return params


# Wrapper classes for backend layers
class Conv2DWrapper(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layer = backend.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class LinearWrapper(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = backend.Linear(in_features, out_features, bias)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class ReLUWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.ReLU()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class MaxPool2DWrapper(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer = backend.MaxPool2D(kernel_size, stride)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class BatchNorm2DWrapper(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.layer = backend.BatchNorm2D(num_features, eps, momentum)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class BatchNorm1DWrapper(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.layer = backend.BatchNorm1D(num_features, eps, momentum)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class DropoutWrapper(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.layer = backend.Dropout(p)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class FlattenWrapper(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.layer = backend.Flatten(start_dim, end_dim)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class LeakyReLUWrapper(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.layer = backend.LeakyReLU(negative_slope)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class TanhWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.Tanh()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class SigmoidWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.Sigmoid()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class AvgPool2DWrapper(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer = backend.AvgPool2D(kernel_size, stride)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class GlobalAvgPool2DWrapper(Module):
    """Global Average Pooling - averages each feature map to a single value.
    Input:  [batch, channels, H, W]
    Output: [batch, channels, 1, 1] -> flattened to [batch, channels]
    """
    def __init__(self):
        super().__init__()
        self._pool_layer = None
        self._flatten = backend.Flatten(1, -1)
    
    def forward(self, x):
        # Get spatial size dynamically
        h = x.shape[2]
        w = x.shape[3]
        # Use AvgPool2D with kernel = spatial size
        self._pool_layer = backend.AvgPool2D(h, 1)
        out = self._pool_layer.forward(x)
        # Flatten [batch, C, 1, 1] -> [batch, C]
        out = self._flatten.forward(out)
        return out
    
    def backward(self, grad):
        # Unflatten grad back
        grad = self._flatten.backward(grad)
        return self._pool_layer.backward(grad)


class ResidualBlockWrapper(Module):
    """Residual block with skip connection.
    
    Main path:  Conv3x3(in→out, stride) → BN → ReLU → Conv3x3(out→out) → BN
    Shortcut:   1x1 Conv(in→out, stride) → BN  (if dimensions change)
                Identity                         (if dimensions match)
    Output:     ReLU(main + shortcut)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.need_shortcut = (stride != 1 or in_channels != out_channels)
        
        # Main path
        self.conv1 = backend.Conv2D(in_channels, out_channels, 3, stride, 1, False)  # 3x3, pad=1, no bias
        self.bn1 = backend.BatchNorm2D(out_channels)
        self.relu1 = backend.ReLU()
        self.conv2 = backend.Conv2D(out_channels, out_channels, 3, 1, 1, False)  # 3x3, stride=1, pad=1, no bias
        self.bn2 = backend.BatchNorm2D(out_channels)
        
        # Shortcut (1x1 conv + BN if dimensions change)
        if self.need_shortcut:
            self.shortcut_conv = backend.Conv2D(in_channels, out_channels, 1, stride, 0, False)
            self.shortcut_bn = backend.BatchNorm2D(out_channels)
        
        # Final ReLU
        self.relu_out = backend.ReLU()
        
        # Collect parameters
        self._parameters = []
        self._parameters.extend(self.conv1.parameters())
        self._parameters.extend(self.bn1.parameters())
        self._parameters.extend(self.conv2.parameters())
        self._parameters.extend(self.bn2.parameters())
        if self.need_shortcut:
            self._parameters.extend(self.shortcut_conv.parameters())
            self._parameters.extend(self.shortcut_bn.parameters())
        
        # Cache for backward
        self._input = None
        self._main_out = None
        self._shortcut_out = None
    
    def forward(self, x):
        self._input = x
        
        # Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        self._main_out = out
        
        # Shortcut path
        if self.need_shortcut:
            shortcut = self.shortcut_conv.forward(x)
            shortcut = self.shortcut_bn.forward(shortcut)
        else:
            shortcut = x
        self._shortcut_out = shortcut
        
        # Add and ReLU
        result = out + shortcut
        result = self.relu_out.forward(result)
        return result
    
    def backward(self, grad):
        # Backward through final ReLU
        grad = self.relu_out.backward(grad)
        
        # Gradient splits into main path and shortcut path
        main_grad = grad  # grad flows to both branches
        shortcut_grad = grad
        
        # Backward through main path: bn2 -> conv2 -> relu1 -> bn1 -> conv1
        main_grad = self.bn2.backward(main_grad)
        main_grad = self.conv2.backward(main_grad)
        main_grad = self.relu1.backward(main_grad)
        main_grad = self.bn1.backward(main_grad)
        main_grad = self.conv1.backward(main_grad)
        
        # Backward through shortcut
        if self.need_shortcut:
            shortcut_grad = self.shortcut_bn.backward(shortcut_grad)
            shortcut_grad = self.shortcut_conv.backward(shortcut_grad)
        
        # Sum gradients from both paths (dL/dx = dL/dx_main + dL/dx_shortcut)
        input_grad = main_grad + shortcut_grad
        return input_grad
    
    def train(self):
        self._training = True
        self.bn1.train()
        self.bn2.train()
        if self.need_shortcut:
            self.shortcut_bn.train()
    
    def eval(self):
        self._training = False
        self.bn1.eval()
        self.bn2.eval()
        if self.need_shortcut:
            self.shortcut_bn.eval()
