import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    DEVICE, IMG_SIZE, INCEPTION_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, seed_everything, SEED
)
from models import create_model
from train import evaluate

def get_test_dataloader(model_name: str, data_dir: str, batch_size: int = 32):
    """Create a dataloader for the test set."""
    size = INCEPTION_SIZE if model_name == "inception_v3" else IMG_SIZE
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    # We assume data_dir contains subdirectories for each class, 
    # similar to ImageFolder format.
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=(DEVICE.type == "cuda")
        )
        return loader
    except Exception as e:
        print(f"Error loading test data from {data_dir}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a test set.")
    parser.add_argument("--model", type=str, required=True, help="Model architecture name")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weight file")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    args = parser.parse_args()

    seed_everything(SEED)
    
    print(f"\nEvaluating {args.model} on test set...")
    print(f"Weights: {args.weights}")
    print(f"Test Dir: {args.test_dir}")

    if not os.path.exists(args.weights):
        print(f"ERROR: Weight file not found: {args.weights}")
        return

    if not os.path.isdir(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        return

    # 1. Load data
    loader = get_test_dataloader(args.model, args.test_dir, args.batch_size)
    if loader is None or len(loader) == 0:
        print("ERROR: Could not create dataloader or test set is empty.")
        return

    # 2. Setup model
    try:
        model = create_model(args.model).to(DEVICE)
        state_dict = torch.load(args.weights, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR: Failed to load model or weights: {e}")
        return

    # 3. Evaluate
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion)

    print("-" * 30)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Acc : {acc*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()
