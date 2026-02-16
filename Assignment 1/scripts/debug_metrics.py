
import sys
import os
from pathlib import Path

# Add project root and build to path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'build'))

import deepnet_backend as backend
from deepnet.python.module import (Conv2DWrapper, LinearWrapper, ReLUWrapper, 
                                  MaxPool2DWrapper, BatchNorm2DWrapper, 
                                  FlattenWrapper, Sequential)
from utils.metrics import count_parameters, estimate_macs_flops

def test_metrics():
    # Conv2D
    conv = Conv2DWrapper(3, 16, 3, stride=1, padding=1)
    print(f"Conv2D p: {count_parameters(conv)} | s: {getattr(conv, 'stride', None)} | p: {getattr(conv, 'padding', None)}")
    
    # BatchNorm2D
    bn = BatchNorm2DWrapper(16)
    print(f"BN p: {count_parameters(bn)}") # Should be 32 now
    
    # MaxPool2D
    pool = MaxPool2DWrapper(2, 2)
    print(f"Pool s: {getattr(pool, 'stride', None)} | k: {getattr(pool, 'kernel_size', None)}")
    
    # Simple model
    model = Sequential(
        Conv2DWrapper(3, 8, 3, padding=1),
        ReLUWrapper(),
        MaxPool2DWrapper(2, 2),
        FlattenWrapper(),
        LinearWrapper(8 * 16 * 16, 10)
    )
    stats = estimate_macs_flops(model, [1, 3, 32, 32])
    print(f"Model MACs: {stats['macs']:,} | Expected: 241,664")

if __name__ == "__main__":
    test_metrics()

if __name__ == "__main__":
    test_metrics()
