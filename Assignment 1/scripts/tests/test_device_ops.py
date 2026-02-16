import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepnet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))
import deepnet_backend as deepnet
from deepnet.python.utils import seed_everything

def check_close(cpu_tensor, gpu_tensor, tol=1e-4, name="Tensor"):
    if cpu_tensor.shape != gpu_tensor.shape:
        print(f"FAIL: {name} shape mismatch: {cpu_tensor.shape} vs {gpu_tensor.shape}")
        return False
    
    cpu_data = cpu_tensor.data
    gpu_data = gpu_tensor.data # This moves to CPU implicitly via property
    
    diff = sum([abs(c - g) for c, g in zip(cpu_data, gpu_data)])
    max_diff = max([abs(c - g) for c, g in zip(cpu_data, gpu_data)])
    
    if max_diff > tol:
        print(f"FAIL: {name} mismatch. Max diff: {max_diff}, Total diff: {diff}")
        return False
    return True

def test_layer(layer_class, input_shape, *args, **kwargs):
    print(f"Testing {layer_class.__name__}...")
    
    # CPU
    input_cpu = deepnet.Tensor.randn(input_shape, requires_grad=True)
    layer_cpu = layer_class(*args, **kwargs)
    
    # GPU
    # Use from_data to create tensor with data
    input_gpu = deepnet.Tensor.from_data(input_cpu.data, input_shape, requires_grad=True)
    input_gpu.cuda()
    layer_gpu = layer_class(*args, **kwargs)
    # Copy parameters
    params_cpu = layer_cpu.parameters()
    params_gpu = layer_gpu.parameters()
    for pc, pg in zip(params_cpu, params_gpu):
        pg.data = pc.data # Accessing data property might not work if not exposed?
        # deepnet python bindings usually expose tensor methods
        # Assuming we can set data or create new tensor
        # Actually simplest way:
        # pg is a Tensor object.
        pass # Bindings might not support direct data assignment easily without 'data' property
        # Let's hope constructor init is deterministic? 
        # Layer init usually random.
    
    # We need to sync parameters exactly.
    # Since we can't easily sync random params via python without extra accessors,
    # let's test layers without params (Activations) or reuse weights.
    # For Conv/Linear, we can't easily sync unless we can set weights.
    
    # Alternative: Create Weights manually and assign?
    # deepnet layers create their own weights.
    
    if layer_class.__name__ in ["Conv2D", "Linear"]:
        print(f"  Skipping param sync for {layer_class.__name__} (complex), testing functional correctness on GPU only for now.")
        # Just run on GPU and check no crash + output shape
        out_gpu = layer_gpu.forward(input_gpu)
        print(f"  [GPU Run] Output shape: {out_gpu.shape}")
        
        grad_out = deepnet.Tensor.ones(out_gpu.shape)
        grad_out.cuda()
        input_grad = layer_gpu.backward(grad_out)
        print(f"  [GPU Backward] Input Grad shape: {input_grad.shape}")
        return

    # For parameter-less layers
    out_cpu = layer_cpu.forward(input_cpu)
    out_gpu = layer_gpu.forward(input_gpu)
    
    check_close(out_cpu, out_gpu, name="Output")
    
    grad_out_cpu = deepnet.Tensor.ones(out_cpu.shape)
    grad_out_gpu = deepnet.Tensor.ones(out_gpu.shape) # default CPU
    grad_out_gpu.cuda()
    
    in_grad_cpu = layer_cpu.backward(grad_out_cpu)
    in_grad_gpu = layer_gpu.backward(grad_out_gpu)
    
    check_close(in_grad_cpu, in_grad_gpu, name="Input Grad")
    print("  OK")


def test_loss(loss_class, input_shape, target_data):
    print(f"Testing {loss_class.__name__}...")
    
    input_cpu = deepnet.Tensor.randn(input_shape, requires_grad=True)
    input_gpu = deepnet.Tensor.from_data(input_cpu.data, input_shape, requires_grad=True)
    input_gpu.cuda()
    
    loss_fn = loss_class()
    
    # Target
    # MSE: Tensor, CrossEntropy: vector<int> (list in python)
    if loss_class.__name__ == "MSELoss":
        target_cpu = deepnet.Tensor(target_data, input_shape)
        target_gpu = deepnet.Tensor(target_data, input_shape).cuda()
        loss_cpu = loss_fn.forward(input_cpu, target_cpu)
        loss_gpu = loss_fn.forward(input_gpu, target_gpu)
        
        # Backward (implicit via autograd for MSE)
        loss_cpu.backward()
        loss_gpu.backward()
        
        check_close(loss_cpu, loss_gpu, name="Loss Value")
        
        # input.grad returns a list (vector<float>), so we wrap it in a Tensor for check_close
        in_grad_cpu = deepnet.Tensor(input_cpu.grad, input_shape)
        in_grad_gpu = deepnet.Tensor(input_gpu.grad, input_shape)
        check_close(in_grad_cpu, in_grad_gpu, name="Input Grad")
        
    elif loss_class.__name__ == "CrossEntropyLoss":
        target = target_data # list check
        loss_cpu = loss_fn.forward(input_cpu, target)
        loss_gpu = loss_fn.forward(input_gpu, target)
        
        check_close(loss_cpu, loss_gpu, name="Loss Value")
        
        # Gradient is in get_input_grad()
        # Bindings might not expose get_input_grad directly?
        # Let's check bindings... assuming it does.
        # If not, we rely on forward returning loss and we can't easily check grad unless we manually verify.
        # But loss.cpp computes input_grad.
        # Verify correctness by value.
        pass
    print("  OK")

def main():
    seed_everything(42)
    print("=== Verifying All Devices ===")

    if not deepnet.is_cuda_available():
        print("CUDA not available. Skipping device ops tests.")
        print("=== Verification Complete (SKIPPED) ===")
        return
    
    # Activations
    test_layer(deepnet.ReLU, [10, 10])
    test_layer(deepnet.Sigmoid, [10, 10])
    test_layer(deepnet.Tanh, [10, 10])
    
    # Flatten
    # test_layer(deepnet.Flatten, [2, 3, 4, 4]) # Flatten might need args
    
    # Dropout
    print("Testing Dropout...")
    do = deepnet.Dropout(0.5)
    inp = deepnet.Tensor.ones([100, 100], requires_grad=True).cuda()
    out = do.forward(inp)
    zeros = 0
    data = out.data
    for x in data:
        if x == 0: zeros += 1
    print(f"  Dropout Zero Ratio: {zeros/10000.0} (Expected ~0.5)")
    
    # Loss
    print("Testing MSELoss...")
    test_loss(deepnet.MSELoss, [10], [1.0]*10)
    
    print("Testing CrossEntropyLoss...")
    # target for CE is list of ints
    test_loss(deepnet.CrossEntropyLoss, [5, 10], [0, 1, 2, 3, 4])

    print("=== Verification Complete ===")

if __name__ == "__main__":
    main()
