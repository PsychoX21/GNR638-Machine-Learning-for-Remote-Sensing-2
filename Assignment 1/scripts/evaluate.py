#!/usr/bin/env python3
"""
Evaluation script for DeepNet framework
Usage: python scripts/evaluate.py --dataset datasets/data_1 --checkpoint checkpoints/best.pth
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

import deepnet_backend as backend
from deepnet.python.data import ImageFolderDataset, DataLoader, ensure_dataset_extracted
from deepnet.python.models import build_model_from_config, load_checkpoint, calculate_model_stats
from deepnet.python.utils import seed_everything
import time

def flatten_batch(images):
    flat = []
    extend = flat.extend
    for img in images:
        extend(img)
    return flat

def evaluate(model, dataloader, criterion, num_classes, use_cuda=False, channels=3, image_size=32):
    """Evaluate model with per-class metrics"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    from tqdm import tqdm
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    for images, labels in progress_bar:
        # Update description with current accuracy/loss if desired, or just show progress
        # Convert to tensor
        batch_images = flatten_batch(images)
        
        batch_size = len(images)
        input_tensor = backend.Tensor.from_data(
            batch_images,
            [batch_size, channels, image_size, image_size],
            requires_grad=False,
            cuda=use_cuda
        )
        
        outputs = model(input_tensor)
        loss_tensor = criterion.forward(outputs, labels)
        loss = loss_tensor.data[0]
        total_loss += loss
        
        # Get raw data from output tensor
        # Assuming outputs is [batch_size, num_classes]
        out_data = outputs.data
        
        for i in range(batch_size):
            # Extract row for this sample
            row = out_data[i * num_classes : (i + 1) * num_classes]
            max_idx = row.index(max(row))
            true_label = labels[i]
            
            if max_idx == true_label:
                correct += 1
                class_correct[true_label] += 1
            
            total += 1
            class_total[true_label] += 1
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    return avg_loss, accuracy, class_accuracies


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepNet CNN')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to model config YAML (optional, will try to load from checkpoint if not provided)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: use config value)')
    parser.add_argument('--val-split', type=float, default=1.0, help='Fraction of data to use for evaluation (default 1.0 for full test set)')
    
    args = parser.parse_args()
    
    # Set seed for determinism
    seed = 42
    seed_everything(seed)
    print(f"Set random seed: {seed}")
    
    print("=" * 70)
    print("DeepNet Evaluation")
    print("=" * 70)
    
    # 1. Extract config first
    config = None
    if args.config:
        print(f"\nBuilding model from provided config: {args.config}")
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"\nNo config provided, attempting to load from checkpoint: {args.checkpoint}")
        import pickle
        with open(args.checkpoint, 'rb') as f:
            ckpt_data = pickle.load(f)
        config = ckpt_data.get('config')
        if config is None:
            print("Error: No config found in checkpoint and none provided via --config.")
            sys.exit(1)
        print("Successfully loaded model architecture from checkpoint.")

    data_config = config.get('data', {})
    training_config = config.get('training', {})
    image_size = data_config.get('image_size', 32)
    channels = data_config.get('channels', 3)
    
    # Use config batch size if not override
    config_batch_size = training_config.get('batch_size', 64)
    batch_size = args.batch_size if args.batch_size is not None else config_batch_size
    print(f"Batch size: {batch_size}")

    # 2. Load dataset with correct image properties
    print(f"\nLoading dataset: {args.dataset}")
    print(f"  Image properties: {image_size}x{image_size}, {channels} channels")
    dataset_start = time.time()
    
    dataset = ImageFolderDataset(args.dataset, image_size=image_size, channels=channels, train=False, val_split=args.val_split)
    
    dataset_load_time = time.time() - dataset_start
    print(f"Dataset loading time: {dataset_load_time:.2f} seconds")
    
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Build model
    model, built_config = build_model_from_config(config, num_classes)
    
    # Detect and use CUDA if available
    try:
        use_cuda = backend.is_cuda_available()
    except:
        use_cuda = False
    
    print(f"\nCUDA status: {'enabled' if use_cuda else 'disabled (CPU only)'}")
    if use_cuda:
        print("Moving model to CUDA...")
        model.cuda()
    
    # Calculate and print model stats
    stats = calculate_model_stats(
        model, [batch_size, channels, image_size, image_size]
    )
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {stats['parameters']:,}")
    print(f"  MACs: {stats['macs']:,}")
    print(f"  FLOPs: {stats['flops']:,}")
    
    # 4. Load weights
    print(f"\nLoading weights from: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)
    
    criterion = backend.CrossEntropyLoss()
    loss, accuracy, class_accuracies = evaluate(model, dataloader, criterion, num_classes, use_cuda, channels, image_size)
    
    print(f"\nOverall Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print(f"\nPer-Class Accuracy:")
    for class_idx, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {class_accuracies[class_idx]:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
