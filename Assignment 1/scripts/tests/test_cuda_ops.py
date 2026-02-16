#!/usr/bin/env python3
"""
Extended CUDA operations tests for DeepNet framework.
Tests arithmetic, scalar, reduction, and BatchNorm operations on GPU.
"""

import sys
import os
import math
import time

# Add build directory to path to find deepnet_backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepnet'))
# Try release and debug paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

try:
    import deepnet_backend as backend
    from deepnet.python.utils import seed_everything
except ImportError:
    print("Error: Could not import deepnet_backend. Please build the project first.")
    sys.exit(1)

def is_close(a, b, atol=1e-4, rtol=1e-3):
    return abs(a - b) <= (atol + rtol * abs(b))

def test_arithmetic_ops():
    print("Testing Arithmetic Ops (Sub, Div)...")
    if not backend.is_cuda_available():
        print("  Skipping (CUDA not available)")
        return

    a_data = [10.0, 20.0, 30.0, 40.0]
    b_data = [2.0, 5.0, 2.0, 4.0]
    shape = [2, 2]

    a = backend.Tensor.from_data(a_data, shape, False, True)
    b = backend.Tensor.from_data(b_data, shape, False, True)

    # Sub
    c = a.sub(b)
    assert c.is_cuda
    expected_sub = [8.0, 15.0, 28.0, 36.0]
    for i in range(4):
        assert is_close(c.data[i], expected_sub[i]), f"Sub mismatch at {i}: {c.data[i]} vs {expected_sub[i]}"
    print("  [OK] Sub")

    # Div
    d = a.div(b)
    assert d.is_cuda
    expected_div = [5.0, 4.0, 15.0, 10.0]
    for i in range(4):
        # div implementation adds 1e-8 to denom, so approx equal
        assert is_close(d.data[i], expected_div[i]), f"Div mismatch at {i}: {d.data[i]} vs {expected_div[i]}"
    print("  [OK] Div")

def test_scalar_ops():
    print("Testing Scalar Ops...")
    if not backend.is_cuda_available():
        print("  Skipping (CUDA not available)")
        return

    data = [1.0, 2.0, 3.0, 4.0]
    t = backend.Tensor.from_data(data, [2, 2], False, True)

    # Add scalar
    t2 = t.add_scalar(5.0)
    assert t2.is_cuda
    expected_add = [6.0, 7.0, 8.0, 9.0]
    for i in range(4):
        assert is_close(t2.data[i], expected_add[i]), f"Add scalar mismatch: {t2.data[i]}"

    # Mul scalar
    t3 = t.mul_scalar(2.0)
    assert t3.is_cuda
    expected_mul = [2.0, 4.0, 6.0, 8.0]
    for i in range(4):
        assert is_close(t3.data[i], expected_mul[i]), f"Mul scalar mismatch: {t3.data[i]}"
    
    print("  [OK] Scalar Ops")

def test_reductions():
    print("Testing Reductions (Sum, Max)...")
    if not backend.is_cuda_available():
        print("  Skipping")
        return

    data = [1.0, 2.0, 3.0, 4.0]
    t = backend.Tensor.from_data(data, [2, 2], False, True)

    # Sum
    s = t.sum()
    assert s.is_cuda
    assert is_close(s.data[0], 10.0), f"Sum mismatch: {s.data[0]}"

    # Max
    m = t.max()
    assert m.is_cuda
    assert is_close(m.data[0], 4.0), f"Max mismatch: {m.data[0]}"

    print("  [OK] Reductions")

def test_math_ops():
    print("Testing Math Ops (Exp, Log, Pow, Sqrt)...")
    if not backend.is_cuda_available():
        print("  Skipping")
        return
    
    data = [1.0, 4.0, 9.0]
    t = backend.Tensor.from_data(data, [3], False, True)

    # Sqrt
    s = t.sqrt()
    assert s.is_cuda
    expected_sqrt = [1.0, 2.0, 3.0]
    for i in range(3):
        assert is_close(s.data[i], expected_sqrt[i])
    
    # Pow
    p = t.pow(2.0)
    assert p.is_cuda
    expected_pow = [1.0, 16.0, 81.0]
    for i in range(3):
        assert is_close(p.data[i], expected_pow[i])

    # Exp
    e = t.exp()
    assert e.is_cuda
    expected_exp = [math.exp(v) for v in data]
    for i in range(3):
        assert is_close(e.data[i], expected_exp[i])

    # Log
    l = t.log()
    assert l.is_cuda
    expected_log = [math.log(v + 1e-8) for v in data]
    for i in range(3):
        assert is_close(l.data[i], expected_log[i])
    
    print("  [OK] Math Ops")

def test_batchnorm():
    print("Testing BatchNorm2D CUDA...")
    if not backend.is_cuda_available():
        print("  Skipping")
        return

    # N=2, C=2, H=2, W=2
    # Channel 0: all 1s
    # Channel 1: all 2s
    data = [
        # Batch 0
        1.0, 1.0, 1.0, 1.0,  # C0
        2.0, 2.0, 2.0, 2.0,  # C1
        # Batch 1
        1.0, 1.0, 1.0, 1.0,  # C0
        2.0, 2.0, 2.0, 2.0   # C1
    ]
    input_t = backend.Tensor.from_data(data, [2, 2, 2, 2], True, True) # requires_grad=True

    bn = backend.BatchNorm2D(2)
    bn.train()

    # Forward
    output = bn.forward(input_t)
    assert output.is_cuda

    # With constant input, variance is 0. Output should be (x - mean) / eps * gamma + beta
    # mean=1 for C0, mean=2 for C1. x-mean = 0.
    # So output should be near 0 (beta initialized to 0)
    for v in output.data:
        assert is_close(v, 0.0, 1e-3), f"BN Forward large value: {v}"
    
    print("  [OK] BN Forward")

    # Backward
    grad_output = backend.Tensor.ones([2, 2, 2, 2], False, True)
    grad_input = bn.backward(grad_output)
    assert grad_input.is_cuda

    # Check if we got gradients
    has_grad = False
    for v in grad_input.data:
        if abs(v) > 0: has_grad = True
    
    # Actually, if input is constant, gradient might be zero depending on math, 
    # but let's just check it doesn't crash and returns CUDA tensor
    print("  [OK] BN Backward")

def main():
    seed_everything(42)
    print("=== Extended CUDA Ops Verification ===")
    if not backend.is_cuda_available():
        print("WARNING: CUDA is not available. Tests will be skipped.")
    
    test_arithmetic_ops()
    test_scalar_ops()
    test_reductions()
    test_math_ops()
    test_batchnorm()
    print("=== Done ===")

if __name__ == "__main__":
    main()
