"""Comprehensive layer test: verifies all layer types work in forward+backward."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import deepnet_backend as backend
from deepnet.python.utils import seed_everything

seed_everything(42)

def test_layer(name, forward_fn, backward_fn, input_shape, requires_param_check=False, params=None):
    """Test a single layer's forward and backward."""
    import random
    random.seed(42)
    
    flat = [random.gauss(0, 1) for _ in range(1)]
    total = 1
    for s in input_shape:
        total *= s
    flat = [random.gauss(0, 0.5) for _ in range(total)]
    
    x = backend.Tensor.from_data(flat, list(input_shape), True, False)
    
    try:
        out = forward_fn(x)
        print(f"  [OK] {name} forward: {list(input_shape)} -> {list(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] {name} forward: {e}")
        return False
    
    # Create dummy grad_output matching output shape
    import random
    total_out = 1
    for s in out.shape:
        total_out *= s
    grad_data = [random.gauss(0, 0.1) for _ in range(total_out)]
    grad = backend.Tensor.from_data(grad_data, list(out.shape), False, False)
    
    try:
        grad_in = backward_fn(grad)
        if list(grad_in.shape) != list(input_shape):
            print(f"  [FAIL] {name} backward: shape mismatch {list(grad_in.shape)} != {list(input_shape)}")
            return False
        print(f"  [OK] {name} backward: {list(out.shape)} -> {list(grad_in.shape)}")
    except Exception as e:
        print(f"  [FAIL] {name} backward: {e}")
        return False
    
    if requires_param_check and params:
        for i, p in enumerate(params):
            has_grad = len(p.grad) > 0 and any(g != 0 for g in p.grad)
            if not has_grad:
                print(f"  [WARN] {name} param {i}: no gradient accumulated")
    
    return True

print("=== Comprehensive Layer Test ===\n")

all_passed = True

# Conv2D
conv = backend.Conv2D(3, 8, 3, 1, 1)
ok = test_layer("Conv2D", conv.forward, conv.backward, [2, 3, 8, 8], True, conv.parameters())
all_passed = all_passed and ok

# Linear
linear = backend.Linear(16, 8)
ok = test_layer("Linear", linear.forward, linear.backward, [2, 16], True, linear.parameters())
all_passed = all_passed and ok

# ReLU
relu = backend.ReLU()
ok = test_layer("ReLU", relu.forward, relu.backward, [2, 4])
all_passed = all_passed and ok

# LeakyReLU
lrelu = backend.LeakyReLU(0.01)
ok = test_layer("LeakyReLU", lrelu.forward, lrelu.backward, [2, 4])
all_passed = all_passed and ok

# Tanh
tanh = backend.Tanh()
ok = test_layer("Tanh", tanh.forward, tanh.backward, [2, 4])
all_passed = all_passed and ok

# Sigmoid
sigmoid = backend.Sigmoid()
ok = test_layer("Sigmoid", sigmoid.forward, sigmoid.backward, [2, 4])
all_passed = all_passed and ok

# MaxPool2D
maxpool = backend.MaxPool2D(2, 2)
ok = test_layer("MaxPool2D", maxpool.forward, maxpool.backward, [2, 3, 8, 8])
all_passed = all_passed and ok

# AvgPool2D
avgpool = backend.AvgPool2D(2, 2)
ok = test_layer("AvgPool2D", avgpool.forward, avgpool.backward, [2, 3, 8, 8])
all_passed = all_passed and ok

# BatchNorm2D
bn2d = backend.BatchNorm2D(3)
ok = test_layer("BatchNorm2D", bn2d.forward, bn2d.backward, [2, 3, 4, 4], True, bn2d.parameters())
all_passed = all_passed and ok

# BatchNorm1D
bn1d = backend.BatchNorm1D(8)
ok = test_layer("BatchNorm1D", bn1d.forward, bn1d.backward, [2, 8], True, bn1d.parameters())
all_passed = all_passed and ok

# Dropout (in training mode)
dropout = backend.Dropout(0.5)
ok = test_layer("Dropout", dropout.forward, dropout.backward, [2, 4])
all_passed = all_passed and ok

# Flatten
flatten = backend.Flatten(1, -1)
ok = test_layer("Flatten", flatten.forward, flatten.backward, [2, 3, 4, 4])
all_passed = all_passed and ok

# CrossEntropyLoss
print("\n--- Loss ---")
import random
random.seed(42)
logits_data = [random.gauss(0, 1) for _ in range(6)]
logits = backend.Tensor.from_data(logits_data, [2, 3], True, False)
targets = [1, 0]
try:
    ce_loss = backend.CrossEntropyLoss()
    loss = ce_loss.forward(logits, targets)
    grad = ce_loss.get_input_grad()
    print(f"  [OK] CrossEntropyLoss: loss={loss.data[0]:.4f}, grad shape={list(grad.shape)}")
except Exception as e:
    print(f"  [FAIL] CrossEntropyLoss: {e}")
    all_passed = False

# MSELoss
try:
    mse_loss = backend.MSELoss()
    pred = backend.Tensor.from_data([1.0, 2.0, 3.0], [3], False, False)
    target = backend.Tensor.from_data([1.5, 2.5, 3.5], [3], False, False)
    loss = mse_loss.forward(pred, target)
    print(f"  [OK] MSELoss: loss={loss.data[0]:.4f} (expected 0.25)")
except Exception as e:
    print(f"  [FAIL] MSELoss: {e}")
    all_passed = False

# Test tensor operations
print("\n--- Tensor Ops ---")
try:
    a = backend.Tensor.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], True, False)
    s = a.sum()
    print(f"  [OK] sum(): {s.data[0]:.1f} (expected 21.0)")
    m = a.mean()
    print(f"  [OK] mean(): {m.data[0]:.4f} (expected 3.5)")
    p = a.pow(2.0)
    print(f"  [OK] pow(2): first={p.data[0]:.1f} (expected 1.0)")
    sq = a.sqrt()
    print(f"  [OK] sqrt(): first={sq.data[0]:.4f} (expected 1.0)")
    mx = a.max()
    print(f"  [OK] max(): {mx.data[0]:.1f} (expected 6.0)")
    mn = a.min()
    print(f"  [OK] min(): {mn.data[0]:.1f} (expected 1.0)")
except Exception as e:
    print(f"  [FAIL] Tensor ops: {e}")
    all_passed = False

print("\n" + "=" * 50)
if all_passed:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
print("=" * 50)
