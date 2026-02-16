"""Utilities for calculating model metrics"""

def count_parameters(model):
    """Count total trainable parameters"""
    total = 0
    params = model.parameters()
    for param in params:
        if getattr(param, 'requires_grad', True):
            param_count = 1
            for dim in param.shape:
                param_count *= dim
            total += param_count
    return total


def estimate_macs_flops(model, input_shape):
    """
    Estimate MACs and FLOPs for the model
    
    Args:
        model: The neural network model
        input_shape: Input tensor shape [batch, channels, height, width]
    
    Returns:
        dict: Dictionary with 'macs' and 'flops' estimates
    """
    macs = 0
    flops = 0
    
    # Get model layers
    if hasattr(model, 'layers'):
        layers = model.layers
    else:
        return {'macs': 0, 'flops': 0}
    
    current_shape = list(input_shape)
    
    for layer in layers:
        layer_type = type(layer).__name__
        
        if 'Conv2D' in layer_type:
            # Conv2D: (batch * out_h * out_w * out_c) * (kernel_h * kernel_w * in_c)
            if hasattr(layer, 'layer'):
                backend_layer = layer.layer
                params = backend_layer.parameters()
                if params:
                    weight = params[0]
                    # Weight shape: [out_c, in_c, kernel_h, kernel_w]
                    out_c = weight.shape[0]
                    in_c = weight.shape[1]
                    kernel_h = weight.shape[2]
                    kernel_w = weight.shape[3]
                    
                    # Get stride (default 1) and padding (default 0 or 1 usually)
                    stride = getattr(layer, 'stride', 1)
                    padding = getattr(layer, 'padding', 0)
                    if isinstance(padding, tuple): padding = padding[0] # Handle tuple padding
                    
                    # Calculate output dimensions
                    # H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
                    if len(current_shape) == 4:
                        out_h = (current_shape[2] + 2 * padding - kernel_h) // stride + 1
                        out_w = (current_shape[3] + 2 * padding - kernel_w) // stride + 1
                        
                        layer_macs = (current_shape[0] * out_h * out_w * out_c) * (kernel_h * kernel_w * in_c)
                        macs += layer_macs
                        
                        # DEBUG
                        # print(f"DEBUG: Conv2D {in_c}->{out_c} k={kernel_h} s={stride} p={padding} | In: {current_shape} -> Out: {out_h}x{out_w} | MACs: {layer_macs:,.0f}")
                        
                        # Update current shape
                        current_shape[1] = out_c
                        current_shape[2] = int(out_h)
                        current_shape[3] = int(out_w)
        
        elif 'ResidualBlock' in layer_type:
            # ResidualBlock: Conv1 -> BN -> ReLU -> Conv2 -> BN (+ Shortcut)
            # We simulate the internal layers
            in_c = layer.in_channels
            out_c = layer.out_channels
            stride = layer.stride
            
            if len(current_shape) == 4:
                # Conv1: 3x3, stride=stride, padding=1
                # Weight: [out_c, in_c, 3, 3]
                current_h = current_shape[2]
                current_w = current_shape[3]
                
                # Conv1 Output
                out_h = (current_h + 2*1 - 3) // stride + 1
                out_w = (current_w + 2*1 - 3) // stride + 1
                conv1_macs = (current_shape[0] * out_h * out_w * out_c) * (3 * 3 * in_c)
                macs += conv1_macs
                
                # Conv2: 3x3, stride=1, padding=1
                # Input is [batch, out_c, out_h, out_w]
                # Output is same size
                conv2_macs = (current_shape[0] * out_h * out_w * out_c) * (3 * 3 * out_c)
                macs += conv2_macs
                
                # Shortcut: 1x1 conv if dimensions change
                if stride != 1 or in_c != out_c:
                    # Shortcut Conv: 1x1, stride=stride, padding=0
                    # Output size matches Conv1 output
                    shortcut_macs = (current_shape[0] * out_h * out_w * out_c) * (1 * 1 * in_c)
                    macs += shortcut_macs
                
                # Update shape to output of block
                current_shape[1] = out_c
                current_shape[2] = int(out_h)
                current_shape[3] = int(out_w)
        
        elif 'Linear' in layer_type:
            # Linear: batch * in_features * out_features
            if hasattr(layer, 'layer'):
                backend_layer = layer.layer
                params = backend_layer.parameters()
                if params:
                    weight = params[0]
                    # Weight shape: [out_features, in_features]
                    out_features = weight.shape[0]
                    in_features = weight.shape[1]
                    
                    batch = current_shape[0]
                    layer_macs = batch * in_features * out_features
                    macs += layer_macs
                    
                    # Update shape
                    current_shape = [batch, out_features]
                    
        elif 'GlobalAvgPool' in layer_type:
            # Output is [batch, channels] (flattened)
            if len(current_shape) == 4:
                current_shape = [current_shape[0], current_shape[1]]

        elif 'MaxPool' in layer_type or 'AvgPool' in layer_type:
            # Pooling reduces spatial dimensions
            # Assuming kernel_size = 2, stride = 2 if not specified
            # But better to check layer attributes if possible
            if len(current_shape) == 4:
                # Basic assumption if attributes unavailable
                k = getattr(layer, 'kernel_size', 2)
                s = getattr(layer, 'stride', 2)
                if isinstance(k, tuple): k = k[0]
                if isinstance(s, tuple): s = s[0]
                
                current_shape[2] = (current_shape[2] - k) // s + 1
                current_shape[3] = (current_shape[3] - k) // s + 1
        
        elif 'Flatten' in layer_type:
            # Flatten to 2D
            if len(current_shape) == 4:
                flat_size = current_shape[1] * current_shape[2] * current_shape[3]
                current_shape = [current_shape[0], flat_size]
    
    # FLOPs are approximately 2 * MACs (multiply + add)
    flops = 2 * macs
    
    return {'macs': macs, 'flops': flops}


def print_model_summary(model, input_shape):
    """Print a summary of the model"""
    print("\nModel Summary")
    print("=" * 60)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    
    # Estimate MACs and FLOPs
    metrics = estimate_macs_flops(model, input_shape)
    print(f"Estimated MACs: {metrics['macs']:,}")
    print(f"Estimated FLOPs: {metrics['flops']:,}")
    
    # Model size in MB (assuming float32)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    print("=" * 60)
