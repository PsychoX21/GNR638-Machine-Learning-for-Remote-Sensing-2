# DeepNet: Custom Deep Learning Framework

**GNR638 Machine Learning for Remote Sensing — Assignment 1**

A high-performance CNN framework built from scratch with a C++ backend and Python frontend. Implements all tensor operations, layers, optimizers, and training utilities without any external ML libraries. Includes OpenMP for multi-threaded CPU parallelization and optional CUDA for GPU acceleration.

> [**IMPORTANT**]
> **Performance Note:** This codebase has been optimized and verified on an **NVIDIA RTX 4070 Laptop GPU** using CUDA and OpenMP. On this hardware, training completes in **< 3 hours** and evaluation in **< 1 hour** per dataset.
>
> While CPU-only execution is fully supported and verified (via OpenMP), it is **significantly slower**. We strongly recommend using a CUDA-enabled NVIDIA GPU for reasonable training times. If you must run on CPU, please expect extended execution times.

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Notes |
| --- | --- | --- |
| Python | 3.10+ | 3.12+ recommended |
| CMake | 3.15+ | [cmake.org/download](https://cmake.org/download/) |
| Git | Any | For cloning pybind11 |
| C++ Compiler | C++17 support | See platform-specific below |
| OpenMP | Optional | Auto-detected, speeds up CPU |
| CUDA Toolkit | Optional (11.0+) | Auto-detected, enables GPU |

**Platform-specific compilers:**

- **Windows**: Visual Studio 2019+ with "Desktop development with C++" workload
- **Linux**: `sudo apt install build-essential cmake python3-dev`
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

### 1. Build Framework

```bash
# 1. Setup environment (first time only)
make setup

# 2. Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\activate
# Windows CMD:
venv\Scripts\activate

# 3. Build C++ backend and install
make build install

# 4. Verify everything works
make test
```

### 2. Run Training and Evaluation

```bash
# MNIST (10 digits)
make train DATA=datasets/data_1 CONFIG=configs/mnist.yaml
make eval DATA=datasets/data_1 MODEL=checkpoints/best_data_1.pth

# CIFAR-100 (100 classes)
make train DATA=datasets/data_2 CONFIG=configs/cifar100.yaml
make eval DATA=datasets/data_2 MODEL=checkpoints/best_data_2.pth
```

---

## Makefile Summary

| Target | Description |
| --- | --- |
| `make` | Full setup + build + install |
| `make setup` | Create venv, install deps, clone pybind11 |
| `make build` | Compile C++ backend with CMake |
| `make install` | Copy compiled module and install Python package |
| `make test` | Run all tests (layers + gradient + CUDA) |
| `make test-layers` | Run layer/tensor operation tests |
| `make test-gradient` | Run gradient and training convergence test |
| `make test-cuda` | Run CUDA GPU acceleration tests |
| `make train` | Train model (`DATA=`, `CONFIG=`, `EPOCHS=`, `BATCH_SIZE=`) |
| `make eval` | Evaluate model (`DATA=`, `MODEL=`) |
| `make clean` | Remove build artifacts |
| `make distclean` | Deep clean (also removes venv, pybind11) |

---

## Project Structure

```
GNR-Assignment-1/
├── Makefile                        # Cross-platform build system
├── CMakeLists.txt                  # CMake configuration
├── setup.py                        # Python package installer
├── requirements.txt                # Python dependencies
│
├── deepnet/                        # Core Framework
│   ├── cpp/                        # C++ Backend (High Performance)
│   │   ├── include/                # Headers (Tensor, Layers, Optimizers, CUDA)
│   │   └── src/                    # Implementations (OpenMP & CUDA kernels)
│   ├── bindings/                   # pybind11 Python bindings
│   └── python/                     # Python Wrapper
│       ├── data.py                 # Dataset loading & augmentation
│       ├── module.py               # Module & Sequential abstractions
│       └── models.py               # Config builder & statistics
│
├── scripts/                        # Entry points
│   ├── train.py                    # Training engine
│   ├── evaluate.py                 # Self-contained evaluator
│   └── tests/                      # Extensive test suite
│       ├── test_all_layers.py      # Layer correctness
│       ├── test_gradient.py        # Gradient verification
│       ├── test_determinism.py     # Seeding verification
│       └── test_cuda.py            # GPU acceleration
│
├── configs/                        # YAML Model Definitions
│   ├── mnist.yaml                  # Optimized for data_1
│   └── cifar100.yaml               # Optimized for data_2
│
├── utils/                          # Helper utilities
│   ├── metrics.py                  # Parameters, MACs, FLOPs calculation
│   └── visualization.py            # Logging & Progress
│
└── report/                       # Report Files
    ├── report.pdf                  # Compiled PDF
    └── report.tex                  # LaTeX file for report
```

---

## Training

The training script dynamically loads model architecture and hyperparameters from YAML configs.

```bash
# MNIST (data_1)
python scripts/train.py --dataset datasets/data_1 --config configs/mnist.yaml

# CIFAR-100 (data_2)
python scripts/train.py --dataset datasets/data_2 --config configs/cifar100.yaml
```

| Argument | Description |
| --- | --- |
| `--dataset` | Path to dataset directory |
| `--config` | Path to model config YAML (provides baseline defaults) |
| `--epochs` | Override number of training epochs |
| `--batch-size` | Override batch size |
| `--val-split` | Train/validation split ratio (default: 0.1) |
| `--checkpoint-dir` | Where to save model checkpoints |

> [!NOTE]
> Command-line arguments (like `--batch-size`) will **overwrite** the values specified in the YAML configuration.

**Outputs:**

- `checkpoints/best_data_1.pth` — Best validation accuracy model (per dataset)
- `checkpoints/data_1_epoch_10.pth` — Periodic checkpoints (every 10 epochs)
- Console: per-epoch train/val loss, accuracy, timing

**CUDA:** If an NVIDIA GPU is detected, tensor operations automatically dispatch to GPU kernels. No code changes needed — the script prints `CUDA status: enabled` at startup.

---

## Visualization

The training script automatically creates log files in the `logs/` directory (e.g., `logs/train_data_1_mnist_20260216.log`). You can generate training curves from these logs using the provided utility:

```bash
# General usage
python utils/plot_logs.py logs/your_log_file.log --name "Model Name"

# Example
python utils/plot_logs.py logs/train_data_1_mnist.log --name "MNIST ResNet"
```

This will save `_loss.png` and `_acc.png` plots to the `report/images/` directory.

---

## Standalone Evaluation

The evaluation script is designed to be fully self-contained. It stores the model architecture configuration inside the `.pth` checkpoint file during training. This allows graders to run evaluation without needing the original config file.

```bash
# Grader evaluation command
python scripts/evaluate.py --dataset [dataset_dir] --checkpoint [model.pth]
```

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | (required) | Path to dataset directory |
| `--checkpoint` | (required) | Path to `.pth` model file |
| `--config` | (None) | Optional manual config override |
| `--batch-size` | 64 | Batch size |
| `--val-split` | 1.0 | Fraction of data to use (1.0 = all) |

Prints overall accuracy, loss, and per-class accuracy breakdown.

---

## Model Configuration

Models are defined in YAML config files, allowing for rapid experimentation without code changes.

| Config | Description |
| --- | --- |
| `mnist.yaml` | High-accuracy ResNet-style architecture for 28x28 grayscale |
| `cifar100.yaml` | Deeper CNN with residual connections for 32x32 RGB |

The final layer uses `out_features: "num_classes"` which is automatically replaced based on the dataset.

### Config Format Example

```yaml
model:
  name: "DeepResNet"
  architecture:
    - type: "Conv2D"
      in_channels: 1
      out_channels: 16
      kernel_size: 3
      stride: 1
      padding: 1
    - type: "ResidualBlock"
      in_channels: 16
      out_channels: 16
    - type: "GlobalAveragePooling2D"
    - type: "Linear"
      in_features: 16
      out_features: "num_classes"  # Automatically set based on dataset

training:
  optimizer: "SGD"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  batch_size: 128
  epochs: 40

data:
  image_size: 32
  channels: 1

device:
  use_cuda: true
```

---

## Technical Highlights

### High Performance & Acceleration

- **OpenMP CPU Parallelization**: Multi-threaded tensor operations and layer computations.
- **CUDA GPU Acceleration**: Optimized kernels for matrix multiplication and activations.
- **C++17 Backend**: Pure C++ core with zero NumPy/PyTorch dependencies.

### Framework Features

- **Layers**: Conv2D, Residual Blocks, BatchNorm, Pooling (Max/Avg), Dropout, Linear.
- **Activations**: ReLU, LeakyReLU, Sigmoid, Tanh.
- **Optimizers**: Adam and SGD (with momentum and weight decay).
- **Self-Contained**: Automatic architecture reconstruction from `.pth` metadata.

---

## Design Insights (For Report)

- **Input Invariance**: The architecture uses **Global Average Pooling**, making it robust to different input spatial dimensions (e.g., 28x28 or 32x32) without changing the Linear layer parameters.
- **Efficiency**: Image loading uses a **Hybrid Preloading** strategy—images are resized once and stored in RAM, while heavy augmentations (if enabled) are applied on-the-fly to balance memory and speed.
- **Stability**: Implemented **He Initialization** logic within the C++ layer constructors to ensure stable gradient flow from the first epoch.

---

## Sources & References

- **pybind11**: Used for C++/Python interoperability.
- **OpenCV**: Used exclusively for image I/O and basic resizing.
- **PyYAML**: Used for parsing model configuration files.
- **Tqdm**: Used for training progress visualization.
- **Automatic Differentiation**: Implementation logic inspired by the principles of Computational Graphs and Backpropagation.

---

## Assignment Compliance

- **Zero External ML Libraries**: Entirely custom implementation (No NumPy/PyTorch/SciPy).
- **C++ Backend (+20 Bonus)**: Full performance core with Python bindings.
- **Standalone Evaluation**: Model configuration embedded in weight files.
- **Complexity Metrics**: Accurate Parameters, MACs, and FLOPs reporting.
- **Dataset Metrics**: Measured and reported loading time for all datasets.
- **Time Constraints**: Training completes well under the 3-hour limit per dataset.
- **Cross-Platform**: Verified on Windows (MSVC), Linux (GCC), and macOS.

---

## License

Educational project for GNR638 coursework.
