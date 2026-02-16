#!/usr/bin/env python3
"""
CUDA GPU acceleration tests for DeepNet framework.
Tests all CUDA-accelerated tensor operations and verifies correctness.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepnet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import deepnet_backend as backend
from deepnet.python.utils import seed_everything


def test_cuda_availability():
    """Test CUDA runtime detection."""
    available = backend.is_cuda_available()
    print(f"  CUDA available: {available}")
    return available


def test_tensor_cuda_flag():
    """Test is_cuda flag on tensor creation and transfer."""
    # CPU tensor
    cpu_t = backend.Tensor.from_data([1, 2, 3, 4], [2, 2], False, False)
    assert not cpu_t.is_cuda, "CPU tensor should have is_cuda=False"

    # CUDA tensor via from_data
    gpu_t = backend.Tensor.from_data([1, 2, 3, 4], [2, 2], False, True)
    assert gpu_t.is_cuda, "CUDA tensor should have is_cuda=True"

    # .cuda() method
    cpu_t.cuda()
    assert cpu_t.is_cuda, ".cuda() should set is_cuda=True"

    # .cpu() method
    cpu_t.cpu()
    assert not cpu_t.is_cuda, ".cpu() should set is_cuda=False"

    # .to() method
    t2 = gpu_t.to(False)
    assert not t2.is_cuda, ".to(False) should return CPU tensor"
    t3 = t2.to(True)
    assert t3.is_cuda, ".to(True) should return CUDA tensor"

    # Factory methods
    z = backend.Tensor.zeros([3, 3], False, True)
    assert z.is_cuda, "Tensor.zeros with cuda=True should set flag"

    o = backend.Tensor.ones([3, 3], False, True)
    assert o.is_cuda, "Tensor.ones with cuda=True should set flag"

    r = backend.Tensor.randn([3, 3], 0.0, 1.0, False, True)
    assert r.is_cuda, "Tensor.randn with cuda=True should set flag"

    print("  [OK] CUDA flag creation and transfer")


def test_add():
    """Test CUDA element-wise addition."""
    a = backend.Tensor.from_data([1, 2, 3, 4], [2, 2], False, True)
    b = backend.Tensor.from_data([5, 6, 7, 8], [2, 2], False, True)
    c = a.add(b)
    expected = [6.0, 8.0, 10.0, 12.0]
    assert c.is_cuda, "add result should be CUDA"
    assert c.data == expected, f"add: expected {expected}, got {c.data}"
    print(f"  [OK] add: {c.data}")


def test_mul():
    """Test CUDA element-wise multiplication."""
    a = backend.Tensor.from_data([1, 2, 3, 4], [2, 2], False, True)
    b = backend.Tensor.from_data([5, 6, 7, 8], [2, 2], False, True)
    c = a.mul(b)
    expected = [5.0, 12.0, 21.0, 32.0]
    assert c.is_cuda, "mul result should be CUDA"
    assert c.data == expected, f"mul: expected {expected}, got {c.data}"
    print(f"  [OK] mul: {c.data}")


def test_matmul():
    """Test CUDA matrix multiplication."""
    # [2x3] @ [3x2] = [2x2]
    a = backend.Tensor.from_data([1, 2, 3, 4, 5, 6], [2, 3], False, True)
    b = backend.Tensor.from_data([1, 0, 0, 1, 1, 0], [3, 2], False, True)
    c = a.matmul(b)
    expected = [4.0, 2.0, 10.0, 5.0]
    assert c.is_cuda, "matmul result should be CUDA"
    assert c.data == expected, f"matmul: expected {expected}, got {c.data}"
    print(f"  [OK] matmul: {c.data}")


def test_relu():
    """Test CUDA ReLU activation."""
    a = backend.Tensor.from_data([-2, -1, 0, 1, 2, 3], [2, 3], False, True)
    c = a.relu()
    expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    assert c.is_cuda, "relu result should be CUDA"
    assert c.data == expected, f"relu: expected {expected}, got {c.data}"
    print(f"  [OK] relu: {c.data}")


def test_sigmoid():
    """Test CUDA sigmoid activation."""
    a = backend.Tensor.from_data([0, -100, 100], [1, 3], False, True)
    c = a.sigmoid()
    assert c.is_cuda, "sigmoid result should be CUDA"
    assert abs(c.data[0] - 0.5) < 1e-5, f"sigmoid(0) should be ~0.5, got {c.data[0]}"
    assert c.data[1] < 0.01, f"sigmoid(-100) should be ~0, got {c.data[1]}"
    assert c.data[2] > 0.99, f"sigmoid(100) should be ~1, got {c.data[2]}"
    print(f"  [OK] sigmoid: [{c.data[0]:.4f}, {c.data[1]:.6f}, {c.data[2]:.6f}]")


def test_tanh():
    """Test CUDA tanh activation."""
    a = backend.Tensor.from_data([0, -100, 100], [1, 3], False, True)
    c = a.tanh()
    assert c.is_cuda, "tanh result should be CUDA"
    assert abs(c.data[0]) < 1e-5, f"tanh(0) should be ~0, got {c.data[0]}"
    assert abs(c.data[1] + 1.0) < 0.01, f"tanh(-100) should be ~-1, got {c.data[1]}"
    assert abs(c.data[2] - 1.0) < 0.01, f"tanh(100) should be ~1, got {c.data[2]}"
    print(f"  [OK] tanh: [{c.data[0]:.4f}, {c.data[1]:.4f}, {c.data[2]:.4f}]")


def test_flag_propagation():
    """Test that is_cuda flag propagates through chained operations."""
    a = backend.Tensor.from_data([1, -1, 2, -2], [2, 2], False, True)
    b = backend.Tensor.from_data([0.5, 0.5, 0.5, 0.5], [2, 2], False, True)

    r1 = a.relu()
    assert r1.is_cuda, "relu output should propagate is_cuda"

    r2 = r1.add(b)
    assert r2.is_cuda, "add output should propagate is_cuda"

    r3 = r2.sigmoid()
    assert r3.is_cuda, "sigmoid output should propagate is_cuda"

    print("  [OK] Flag propagation through chained ops")


def test_cpu_gpu_consistency():
    """Test that CPU and CUDA produce the same results."""
    data = [1.5, -0.5, 2.0, -1.0, 0.0, 3.0]

    cpu = backend.Tensor.from_data(data, [2, 3], False, False)
    gpu = backend.Tensor.from_data(data, [2, 3], False, True)

    # ReLU
    cpu_r = cpu.relu()
    gpu_r = gpu.relu()
    for i in range(len(data)):
        assert abs(cpu_r.data[i] - gpu_r.data[i]) < 1e-5, \
            f"relu mismatch at {i}: CPU={cpu_r.data[i]}, GPU={gpu_r.data[i]}"

    # Sigmoid
    cpu_s = cpu.sigmoid()
    gpu_s = gpu.sigmoid()
    for i in range(len(data)):
        assert abs(cpu_s.data[i] - gpu_s.data[i]) < 1e-4, \
            f"sigmoid mismatch at {i}: CPU={cpu_s.data[i]}, GPU={gpu_s.data[i]}"

    print("  [OK] CPU/GPU consistency verified")


def test_performance():
    """Simple timing comparison between CPU and GPU for matmul."""
    n = 256
    data = [float(i % 10) for i in range(n * n)]

    cpu_a = backend.Tensor.from_data(data, [n, n], False, False)
    cpu_b = backend.Tensor.from_data(data, [n, n], False, False)

    gpu_a = backend.Tensor.from_data(data, [n, n], False, True)
    gpu_b = backend.Tensor.from_data(data, [n, n], False, True)

    # Warmup
    _ = gpu_a.matmul(gpu_b)

    # CPU timing
    t0 = time.perf_counter()
    for _ in range(3):
        _ = cpu_a.matmul(cpu_b)
    cpu_time = (time.perf_counter() - t0) / 3

    # GPU timing
    t0 = time.perf_counter()
    for _ in range(3):
        _ = gpu_a.matmul(gpu_b)
    gpu_time = (time.perf_counter() - t0) / 3

    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"  Matmul {n}x{n}: CPU={cpu_time*1000:.1f}ms, GPU={gpu_time*1000:.1f}ms, Speedup={speedup:.1f}x")


def main():
    seed_everything(42)
    print("=" * 50)
    print("=== CUDA GPU Acceleration Tests ===")
    print("=" * 50)

    # Check availability first
    print("\n1. CUDA Availability")
    cuda_available = test_cuda_availability()

    if not cuda_available:
        print("\n  CUDA not available — skipping GPU tests.")
        print("  All tensor ops will run on CPU (this is expected if no NVIDIA GPU).")
        print("\n=== SKIPPED (no GPU) ===")
        return

    print("\n2. Tensor CUDA Flag")
    test_tensor_cuda_flag()

    print("\n3. CUDA Operations (correctness)")
    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_sigmoid()
    test_tanh()

    print("\n4. Flag Propagation")
    test_flag_propagation()

    print("\n5. CPU/GPU Consistency")
    test_cpu_gpu_consistency()

    print("\n6. Performance")
    test_performance()

    print("\n" + "=" * 50)
    print("=== ALL CUDA TESTS PASSED ===")
    print("=" * 50)


if __name__ == '__main__':
    main()
