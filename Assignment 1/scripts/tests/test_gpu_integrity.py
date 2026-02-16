#!/usr/bin/env python3
"""
Expanded GPU Integrity Test: Verifies complex layers (BN, Dropout, Residual, GAP)
and full training loop integration on GPU.
"""

import sys
import os
import time
import random

SEED = 42
random.seed(SEED)

# Add project root and deepnet to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'deepnet'))

try:
    import deepnet_backend as backend
    from deepnet.python.utils import seed_everything
    from deepnet.python.module import Sequential, Conv2DWrapper, LinearWrapper, \
                                       ReLUWrapper, BatchNorm2DWrapper, DropoutWrapper, \
                                       FlattenWrapper, ResidualBlockWrapper, GlobalAvgPool2DWrapper, \
                                       SigmoidWrapper, TanhWrapper
except ImportError as e:
    print(f"Error: Could not import deepnet modules: {e}")
    sys.exit(1)

def check_tensor(t, name, expected_shape=None, should_be_cuda=True):
    """Verify tensor properties."""
    print(f"  Checking {name}...")
    assert t is not None, f"{name} is None"
    
    if expected_shape:
        assert list(t.shape) == list(expected_shape), \
            f"{name} shape mismatch: {list(t.shape)} vs {list(expected_shape)}"
    
    # Check device
    assert t.is_cuda == should_be_cuda, \
        f"{name} device mismatch: is_cuda={t.is_cuda}, expected={should_be_cuda}"
    
    # Check data access (verifies sync logic)
    data = t.data
    assert len(data) > 0, f"{name} data is empty"
    
    # If it was CUDA, it should still be CUDA
    assert t.is_cuda == should_be_cuda, f"{name} device changed after data access"

def test_gpu_pipeline():
    seed_everything(42)
    print("\n=== Testing Complex GPU Pipeline ===")
    
    if not backend.is_cuda_available():
        print("SKIP: CUDA not available")
        return

    # 1. Setup Data (High Res for GAP/Residual)
    print("  1. Creating Input Tensor...")
    # [batch=8, channels=3, H=8, W=8]
    data = [random.gauss(0, 1) for _ in range(8 * 3 * 8 * 8)]
    x = backend.Tensor.from_data(data, [8, 3, 8, 8], True, True)
    check_tensor(x, "Input (x)")

    # 2. Build Complex Model (Mini-ResNet style)
    # Conv(3->8) -> BN -> ReLU -> ResidualBlock(8->16, str=2) -> GAP -> Linear(16->10)
    print("  2. Building Model...")
    model = Sequential(
        Conv2DWrapper(3, 8, 3, padding=1, bias=False),
        BatchNorm2DWrapper(8),
        ReLUWrapper(),
        ResidualBlockWrapper(8, 16, stride=2), # Output: [2, 16, 4, 4]
        GlobalAvgPool2DWrapper(),              # Output: [2, 16]
        DropoutWrapper(p=0.2),
        LinearWrapper(16, 10, bias=True)
    )
    
    # Move model to GPU
    print("  3. Moving Model to GPU...")
    for p in model.parameters():
        p.cuda()
    
    # Check if a few parameters are actually on GPU
    check_tensor(model.parameters()[0], "Model.Weight[0]")

    # 4. Setup Optimizer (Adam) & Loss (CrossEntropy)
    print("  4. Setup Optimizer & Loss...")
    # Lower learning rate slightly for better stability with small batch + Dropout/BN
    optimizer = backend.Adam(model.parameters(), lr=0.002)
    criterion = backend.CrossEntropyLoss()
    
    # 5. Training Step
    print("  5. Starting Training Stage (5 steps to ensure convergence)...")
    model.train()
    
    # Target (8 samples)
    target_list = [3, 7, 1, 9, 2, 5, 0, 4]
    
    # Initial loss (in Eval mode to avoid Dropout jitter)
    print("    Calculating Initial Loss (Eval Mode)...")
    model.eval()
    logits = model.forward(x)
    loss = criterion.forward(logits, target_list)
    initial_loss = loss.data[0]
    print(f"    Initial Loss: {initial_loss:.4f}")

    # 5. Training Steps
    print("  5. Starting Training Stage (20 steps)...")
    model.train()
    for step in range(20):
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = criterion.forward(logits, target_list)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            # Switch to eval to get a stable loss check (ignoring Dropout noise)
            model.eval()
            logits_eval = model.forward(x)
            loss_eval = criterion.forward(logits_eval, target_list)
            print(f"    Step {step}, Loss: {loss_eval.data[0]:.4f} (Eval Mode)")
            model.train()
    
    # 6. Verify Improvement (in Eval mode)
    print("  6. Verifying Weight Update & Loss Decrease (Eval Mode)...")
    model.eval()
    logits_final = model.forward(x)
    loss_final_t = criterion.forward(logits_final, target_list)
    final_loss = loss_final_t.data[0]
    
    print(f"    Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease (Initial: {initial_loss}, Final: {final_loss})"
    print("    Loss successfully decreased over 5 steps")
    # Note: with random data and 1 step, it might not always decrease significantly, 
    # but it shouldn't EXPLODE or CRASH.
    
    # 7. Test Sigmoid and Tanh explicitly
    print("  7. Testing Sigmoid/Tanh on GPU...")
    activations = Sequential(SigmoidWrapper(), TanhWrapper())
    act_out = activations.forward(logits)
    check_tensor(act_out, "Activation Output")
    act_out.backward(backend.Tensor.ones(act_out.shape, False, True))
    print("    Sigmoid/Tanh backward completed")

    print("\n[PASS] GPU Complex Integrity Test Successful")

if __name__ == "__main__":
    test_gpu_pipeline()
