"""Quick gradient check: verifies that the model can learn on synthetic data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Release'))

import deepnet_backend as backend
from deepnet.python.utils import seed_everything

print("=== Quick Gradient Check ===\n")
seed_everything(42)

# Create a tiny 2-layer MLP: Linear(4, 8) -> ReLU -> Linear(8, 3)
linear1 = backend.Linear(4, 8, True)
relu = backend.ReLU()
linear2 = backend.Linear(8, 3, True)

# Collect parameters
params = linear1.parameters() + linear2.parameters()
optimizer = backend.SGD(params, 0.01, 0.0, 0.0)

criterion = backend.CrossEntropyLoss()

import random
random.seed(42)

# Generate synthetic dataset: 100 samples, 4 features, 3 classes
# Simple rule: class = argmax of first 3 features
data = []
labels = []
for _ in range(100):
    x = [random.gauss(0, 1) for _ in range(4)]
    label = max(range(3), key=lambda i: x[i])
    data.append(x)
    labels.append(label)

# Training loop
batch_size = 20
num_epochs = 50
losses = []

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        bs = len(batch_data)
        
        optimizer.zero_grad()
        
        # Flatten batch data
        flat = []
        for x in batch_data:
            flat.extend(x)
        
        input_tensor = backend.Tensor.from_data(flat, [bs, 4], True, False)
        
        # Forward: Linear1 -> ReLU -> Linear2
        h = linear1.forward(input_tensor)
        h = relu.forward(h)
        output = linear2.forward(h)
        
        # Loss
        loss_tensor = criterion.forward(output, batch_labels)
        loss = loss_tensor.data[0]
        total_loss += loss
        
        # Backward through layers
        grad = criterion.get_input_grad()
        if grad is not None:
            grad = linear2.backward(grad)
            grad = relu.backward(grad)
            grad = linear1.backward(grad)
        
        optimizer.step()
        
        # Accuracy
        num_classes = 3
        for j in range(bs):
            pred = max(range(num_classes), key=lambda c: output.data[j * num_classes + c])
            if pred == batch_labels[j]:
                correct += 1
            total += 1
    
    avg_loss = total_loss / (len(data) // batch_size)
    acc = 100.0 * correct / total
    losses.append(avg_loss)
    
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={acc:.1f}%")

print()
# Verify
if losses[-1] < losses[0]:
    print("[PASS] Loss decreased: {:.4f} -> {:.4f}".format(losses[0], losses[-1]))
else:
    print("[FAIL] Loss did NOT decrease: {:.4f} -> {:.4f}".format(losses[0], losses[-1]))

if losses[-1] < 0.5:
    print("[PASS] Final loss is low enough (<0.5)")
else:
    print("[WARN] Final loss is still high: {:.4f}".format(losses[-1]))

# Check parameters actually changed
print("\n=== RESULT: ", end="")
if losses[-1] < losses[0] * 0.5:
    print("PASS - Model is learning! ===")
else:
    print("FAIL - Model may not be learning properly ===")
