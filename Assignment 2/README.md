# GNR638: Coding Assignment 2

## Overview

This project evaluates transfer learning, fine-tuning strategies, data efficiency, corruption robustness, and layer-wise feature quality across pre-trained CNN architectures on the Aerial Images Dataset (AID, 30 classes). 

Based on our initial internal benchmarks and requirements, we have selected three core architectures that provide a diverse set of trade-offs between performance, robustness, and architectural design.

### Default Models

| Model | Architecture Type |
|---|---|
| ConvNeXt-Tiny | Modern pure-ConvNet setup |
| Inception v3 | Multi-branch (Inception blocks) |
| DenseNet-121 | Densely-connected layers |

(Note: `resnet50` and `efficientnet_b0` are also completely supported and can be invoked via the CLI or Makefile, but the three models above are the defaults used for the final analysis).

---

## System Requirements & Setup

The code is designed to be easily reproducible with minimum hassle. Running the pipeline requires Python 3.10+ and a CUDA-capable NVIDIA GPU (recommended).

### 1. Create Environment & Install Dependencies

Run the setup command to create a virtual environment (`.venv/`) and automatically install PyTorch, `timm`, and other dependencies.

```bash
# For GPU (CUDA 12.4/12.8 compatible) — recommended for fast training
make ready

# For CPU only — if evaluating on a non-GPU machine
make ready-cpu

# Verify the installation and GPU access
make check
```

### 2. Dataset Setup

Extract the AID dataset following the standard `ImageFolder` structure. You can use any directory and point to it using the `DATA_DIR` override (defaults to `./train_data`).

```text
train_data/
  Airport/
    airport_00001.jpg
    ...
  BareLand/
    ...
  ... (30 classes)
```

### 3. Execution

Once the environment and data are ready, you can create the 80/20 train/validation split and execute the scenarios.

```bash
# Create the train/val split inside a 'results/_split' directory
make setup
```

### 4. Run experiments

```bash
# Run all 5 scenarios with default 3 models
make all

# Run a specific scenario
make scenario1

# Override models, epochs, seed
make scenario1 MODELS=resnet50 EPOCHS=5
make all MODELS=all EPOCHS=10 SEED=123
```

### 5. Final Evaluation (Testing)

After running Scenario 2, the best model weights for the "Full Fine-Tune" strategy (which typically performs best) are automatically saved. You can then evaluate these saved models on an independent test set.

```bash
# Evaluate all default models on a test directory
make test TEST_DIR=./test_data

# Evaluate specific models
make test MODELS="convnext_tiny inception_v3" TEST_DIR=./my_test_set
```

---

## Checking Results

All outputs will be generated exactly inside the `results/` folder (or your `--results-dir` override). This includes:

- **`results.json` files:** Containing quantitative metrics (best validation accuracy achieved, percentage of unfrozen parameters, and execution time).
- **Plot artifacts (.png):** Including confusion matrices, accuracy curves, t-SNE/PCA/UMAP scatter distributions, bar charts showing layer sensitivity norms, and gradient magnitudes.
- **Saved Models (.pth):** For Scenario 2, the weights for the best-performing "Full Fine-Tune" strategy are automatically saved in `results/scenario2/`.

> [!TIP]
> **Detailed Logs:** Efficiency metrics (MAC-FLOPs, parameters) and training/validation losses for every epoch are printed to the console during training. We have provided `training.log` in the root folder as a reference example of the expected output from a full run of all scenarios.

## Makefile Targets

| Target | Description |
|---|---|
| `make ready` | Create venv + install deps (CUDA 12.8) |
| `make ready-cpu` | Create venv + install deps (CPU only) |
| `make check` | Verify PyTorch + CUDA installation |
| `make setup` | Create 80/20 train/val split |
| `make scenario1`..`scenario5` | Run individual scenarios |
| `make all` | Run all 5 scenarios |
| `make test` | Evaluate saved models on test set |
| `make clean` | Delete results directory |

### Makefile Overrides

You can override defaults by appending variables to the `make` command:

```bash
# Override models and epochs
make all MODELS="resnet50 inception_v3" EPOCHS=10

# Override dataset directory and seed
make setup DATA_DIR=./my_custom_data SEED=123

# Custom batch size for specific scenario
make scenario1 BATCH_SIZE=64
```

### Custom Execution (CLI)

If you wish to run a specific model or a single scenario without using Make, simply activate the `.venv` and use the Python CLI:

```bash
# Run specific scenarios with specific models
python main.py --scenario 1 2 --models resnet50 convnext_tiny

# Run all scenarios testing all available models for 10 epochs
python main.py --scenario all --models all --epochs 10

# Override data path
python main.py --scenario 1 --data-dir /path/to/dataset

# Just setup (create split)
python main.py --setup
```

### All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--scenario` | `all` | Scenario(s): `1 2 3 4 5` or `all` |
| `--models` | 3 default | Model name(s) or `all` |
| `--epochs` | `30` | Max epochs (full data) |
| `--epochs-few` | `20` | Max epochs (few-shot) |
| `--data-dir` | `./train_data` | Dataset directory |
| `--results-dir` | `./results` | Output directory |
| `--batch-size` | `32` | Batch size |
| `--seed` | `42` | Random seed |
| `--setup` | — | Only create train/val split |

---

## Project Structure

```
├── config.py          # Global configuration (seeds, device, paths, hyperparams)
├── data.py            # Dataset splitting, transforms, loaders, corruptions
├── models.py          # Model factory, freeze/unfreeze, layer groups, probing
├── train.py           # Training loops (AMP), evaluation, feature extraction
├── visualize.py       # Plotting (curves, CM, embeddings, gradients)
├── scenario1.py       # Linear Probe Transfer
├── scenario2.py       # Fine-Tuning Strategies (saves best weights)
├── scenario3.py       # Few-Shot Learning
├── scenario4.py       # Corruption Robustness
├── scenario5.py       # Layer-Wise Feature Probing
├── test.py            # Testing pipeline (evaluates saved .pth weights)
├── main.py            # CLI entry point
├── Makefile           # Build / run targets (venv-based)
├── README.md          # This file
├── .venv/             # Virtual environment (created by make ready)
├── train_data/        # AID dataset (30 classes, image-folder format)
├── test_data/         # Placeholder for test set
└── results/           # Generated outputs (plots, JSON metrics)
    ├── _split/        #   train/val split
    ├── scenario1/     #   curves, CM, embeddings
    ├── scenario2/     #   grad norms, convergence, acc vs unfrozen, saved weights (.pth)
    ├── scenario3/     #   few-shot accuracy, gap analysis
    ├── scenario4/     #   corruption tables, robustness chart
    └── scenario5/     #   probe accuracy, norms, PCA grid
```

---

## Efficiency Metrics

As noted in the **Checking Results** section, the following are printed during execution and documented in the report, though they are not all persisted in the `results.json` summary files:
- **Number of parameters** (Total, Trainable, and Frozen counts).
- **MACs** (Multiply-Accumulate Operations, calculated via `ptflops`).
- **FLOPs** (Estimated as 2x MACs, shown during training loops).

---

## Reproducibility

All experiments use a fixed random seed (default: 42) applied to:
- Python `random`
- NumPy
- PyTorch (CPU + CUDA)
- cuDNN (deterministic mode)
- `PYTHONHASHSEED`

To reproduce with a different seed:
```bash
make all SEED=123
```

---

---

## Important Notes

### Windows Multi-processing
> [!NOTE]
> The configuration automatically detects if the OS is Windows and sets `NUM_WORKERS=0`. This is to prevent common `PicklingError` or high-overhead issues with PyTorch's `DataLoader` on Windows. This happen seamlessly when using the provided CLI or Makefile.

### Model Precision & Determinism
> [!NOTE]
> Deep learning results, especially with GPU acceleration, can exhibit minor drift between runs due to **CUDA Non-Determinism** and **RNG State Isolation**. 
> Specifically, the current saved model weights for Scenario 2 (Full Fine-Tune) were generated in a separate evaluation pass after the initial benchmarks were recorded. You may notice very slight differences in the "Best Accuracy" reported in the final saved output compared to the original sequential benchmark results. This is expected behavior in these environments.
