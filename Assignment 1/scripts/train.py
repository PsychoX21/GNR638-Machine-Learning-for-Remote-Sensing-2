#!/usr/bin/env python3
"""
Training script for DeepNet framework
Usage: python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 50
"""

import argparse
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

import deepnet_backend as backend
from deepnet.python.data import ImageFolderDataset, DataLoader
from deepnet.python.models import build_model_from_config, calculate_model_stats, save_checkpoint
from deepnet.python.utils import seed_everything
from datetime import datetime

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def flatten_batch(images):
    flat = []
    extend = flat.extend
    for img in images:
        extend(img)
    return flat

# ==============================
# Training
# ==============================

def train_epoch(model, dataloader, criterion, optimizer,
                epoch, use_cuda=False, channels=3, image_size=32):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()

        # Flatten batch
        batch_images = flatten_batch(images)
        batch_size = len(images)

        input_tensor = backend.Tensor.from_data(
            batch_images,
            [batch_size, channels, image_size, image_size],
            requires_grad=False,
            cuda=use_cuda
        )

        # Forward
        outputs = model(input_tensor)

        # Loss
        loss_tensor = criterion.forward(outputs, labels)
        loss = loss_tensor.data[0]

        # Backward
        input_grad = criterion.get_input_grad()
        if input_grad is not None:
            model.backward(input_grad)

        # Update
        optimizer.step()

        # Accuracy
        num_classes = outputs.shape[1]
        logits = outputs.data

        for i in range(batch_size):
            row = logits[i * num_classes:(i + 1) * num_classes]
            pred = row.index(max(row))
            if pred == labels[i]:
                correct += 1
            total += 1

        total_loss += loss

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss:.4f} Acc: {100. * correct / total:.2f}%')

    return total_loss / len(dataloader), 100. * correct / total


# ==============================
# Evaluation
# ==============================

def evaluate(model, dataloader, criterion,
             use_cuda=False, channels=3, image_size=32):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        batch_images = [pixel for img in images for pixel in img]
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

        num_classes = outputs.shape[1]
        logits = outputs.data

        for i in range(batch_size):
            row = logits[i * num_classes:(i + 1) * num_classes]
            pred = row.index(max(row))
            if pred == labels[i]:
                correct += 1
            total += 1

    return total_loss / len(dataloader), 100. * correct / total


# ==============================
# Main
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Train DeepNet CNN')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # Set seed for determinism
    seed = 42
    seed_everything(seed)
    print(f"Set random seed: {seed}")

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Setup file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.dataset).name
    config_name = Path(args.config).stem
    log_file = f"logs/train_{dataset_name}_{config_name}_{timestamp}.log"
    sys.stdout = TeeLogger(log_file)
    print(f"Logging to: {log_file}")

    print("=" * 70)
    print("DeepNet Training")
    print("=" * 70)

    dataset_name = Path(args.dataset).name

    # ------------------------------
    # Load config only
    # ------------------------------
    print(f"\nBuilding model config from: {args.config}")
    _, config = build_model_from_config(args.config, num_classes=0)

    data_config = config.get('data', {})
    training_config = config.get('training', {})

    image_size = data_config.get('image_size', 32)
    channels = data_config.get('channels', 3)
    augmentation = data_config.get('augmentation', {})

    batch_size = args.batch_size or training_config.get('batch_size', 64)
    epochs = training_config.get('epochs', args.epochs)

    print(f"Image size: {image_size}x{image_size}, "
          f"Channels: {channels}, Batch size: {batch_size}, Epochs: {epochs}")

    # ------------------------------
    # Load dataset
    # ------------------------------
    print(f"\nLoading dataset: {args.dataset} ({dataset_name})")
    dataset_start = time.time()

    train_dataset = ImageFolderDataset(
        args.dataset, image_size=image_size, channels=channels,
        train=True, val_split=args.val_split, augmentation=augmentation
    )

    val_dataset = ImageFolderDataset(
        args.dataset, image_size=image_size, channels=channels,
        train=False, val_split=args.val_split
    )

    dataset_load_time = time.time() - dataset_start
    print(f"Dataset loading time: {dataset_load_time:.2f} seconds")

    num_classes = len(train_dataset.classes)

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, "
          f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False)

    # ------------------------------
    # Build final model
    # ------------------------------
    model, config = build_model_from_config(args.config, num_classes)

    stats = calculate_model_stats(
        model, [batch_size, channels, image_size, image_size]
    )

    print(f"\nModel Statistics:")
    print(f"  Parameters: {stats['parameters']:,}")
    print(f"  MACs: {stats['macs']:,}")
    print(f"  FLOPs: {stats['flops']:,}")

    # ------------------------------
    # CUDA
    # ------------------------------
    try:
        use_cuda = backend.is_cuda_available()
    except:
        use_cuda = False

    print(f"\nCUDA status: {'enabled' if use_cuda else 'disabled (CPU only)'}")

    if use_cuda:
        print("Moving model to CUDA...")
        model.cuda()

    # ------------------------------
    # Loss + Optimizer
    # ------------------------------
    criterion = backend.CrossEntropyLoss()
    params = model.parameters()

    optimizer_type = training_config.get('optimizer', 'Adam')
    lr = training_config.get('learning_rate', 0.001)

    if optimizer_type == 'SGD':
        optimizer = backend.SGD(
            params,
            lr=lr,
            momentum=training_config.get('momentum', 0.9),
            weight_decay=training_config.get('weight_decay', 0.0001),
            nesterov=training_config.get('nesterov', False)
        )
    else:
        optimizer = backend.Adam(
            params,
            lr=lr,
            weight_decay=training_config.get('weight_decay', 0.0)
        )

    print(f"\nOptimizer: {optimizer_type}, LR: {lr}")

    # ------------------------------
    # Scheduler
    # ------------------------------
    scheduler = None
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', None)

    if scheduler_type == 'StepLR':
        scheduler = backend.StepLR(
            lr,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
        print(f"Scheduler: StepLR (step={scheduler_config.get('step_size')}, "
              f"gamma={scheduler_config.get('gamma')})")

    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = backend.CosineAnnealingLR(
            lr,
            T_max=scheduler_config.get('T_max', 50),
            eta_min=scheduler_config.get('eta_min', 0.0)
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={scheduler_config.get('T_max')}, "
              f"eta_min={scheduler_config.get('eta_min')})")

    elif scheduler_type == 'ExponentialLR':
        scheduler = backend.ExponentialLR(
            lr,
            gamma=scheduler_config.get('gamma', 0.95)
        )
        print(f"Scheduler: ExponentialLR (gamma={scheduler_config.get('gamma')})")

    else:
        print("Scheduler: None")

    # ------------------------------
    # Training Loop
    # ------------------------------
    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion,
            optimizer, epoch, use_cuda,
            channels, image_size
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion,
            use_cuda, channels, image_size
        )

        epoch_time = time.time() - start

        if scheduler is not None:
            new_lr = scheduler.step()
            optimizer.set_lr(new_lr)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        print(f"{'='*70}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = f'{args.checkpoint_dir}/best_{dataset_name}.pth'
            save_checkpoint(model, best_path, optimizer=optimizer, epoch=epoch, loss=val_loss, config=config)
            print(f"[BEST] Saved best model -> {best_path} "
                  f"(Val Acc: {best_val_acc:.2f}%)")

        if epoch % 10 == 0:
            save_checkpoint(
                model, f'{args.checkpoint_dir}/{dataset_name}_epoch_{epoch}.pth',
                optimizer=optimizer, epoch=epoch, loss=val_loss, config=config
            )

    print("=" * 70)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
