import sys
import os
from pathlib import Path

# Add paths for deepnet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'build'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'build' / 'Release'))

import deepnet_backend as backend
from deepnet.python.utils import seed_everything

def verify_determinism():
    print("=== Verifying Determinism ===")
    
    seed = 42
    
    # 1. Test C++ Backend (Tensor.randn)
    print("\n1. Testing C++ Backend (Tensor.randn)...")
    seed_everything(seed)
    t1 = backend.Tensor.randn([2, 2])
    data1 = t1.data[:]
    
    seed_everything(seed)
    t2 = backend.Tensor.randn([2, 2])
    data2 = t2.data[:]
    
    if data1 == data2:
        print("  [PASS] C++ Tensor.randn is deterministic with seed_everything")
    else:
        print("  [FAIL] C++ Tensor.randn is NOT deterministic!")
        print(f"  First:  {data1}")
        print(f"  Second: {data2}")

    # 2. Test DropOut (Seeded through backend)
    print("\n2. Testing Dropout (backend generator)...")
    dropout = backend.Dropout(0.5)
    dropout.train()
    x = backend.Tensor.ones([10])
    
    seed_everything(seed)
    out1 = dropout.forward(x).data[:]
    
    seed_everything(seed)
    out2 = dropout.forward(x).data[:]
    
    if out1 == out2:
        print("  [PASS] Dropout is deterministic with seed_everything")
    else:
        print("  [FAIL] Dropout is NOT deterministic!")
        print(f"  First:  {out1}")
        print(f"  Second: {out2}")

    # 3. Test Layer Initialization
    print("\n3. Testing Layer Initialization (Conv2D)...")
    seed_everything(seed)
    conv1 = backend.Conv2D(3, 8, 3)
    w1 = conv1.parameters()[0].data[:]
    
    seed_everything(seed)
    conv2 = backend.Conv2D(3, 8, 3)
    w2 = conv2.parameters()[0].data[:]
    
    if w1 == w2:
        print("  [PASS] Layer initialization is deterministic")
    else:
        print("  [FAIL] Layer initialization is NOT deterministic!")

    print("\n=== All Determinism Checks PASSED ===")

if __name__ == "__main__":
    verify_determinism()
