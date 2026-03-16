# GNR638 Assignment 2 Makefile
# Pre-trained CNN Representation Transfer & Robustness Analysis

# Configurable variables
VENV_DIR    ?= .venv
DATA_DIR    ?= ./train_data
TEST_DIR    ?= ./test_data
RESULTS_DIR ?= ./results
SEED        ?= 42
BATCH_SIZE  ?= 32
MODELS      ?= convnext_tiny inception_v3 densenet121
EPOCHS      ?= 30
EPOCHS_FEW  ?= 20

# Platform-adaptive Python / venv paths
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = $(VENV_DIR)/Scripts/python.exe
    ACTIVATE    = $(VENV_DIR)\Scripts\activate
    RM_RF       = if exist $(1) rmdir /s /q $(1)
    MKDIR       = if not exist $(1) mkdir $(1)
    SYS_PYTHON  ?= python
    ECHO_EMPTY  = @echo.
else
    VENV_PYTHON = $(VENV_DIR)/bin/python
    ACTIVATE    = source $(VENV_DIR)/bin/activate
    RM_RF       = rm -rf $(1)
    MKDIR       = mkdir -p $(1)
    SYS_PYTHON  ?= python3
    ECHO_EMPTY  = @echo ""
endif

# Common flags
FLAGS = --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) \
        --seed $(SEED) --batch-size $(BATCH_SIZE) \
        --epochs $(EPOCHS) --epochs-few $(EPOCHS_FEW) \
        --models $(MODELS)

.PHONY: help ready ready-cpu setup all scenario1 scenario2 scenario3 scenario4 scenario5 clean check test

# Default target
help:
	@echo "============================================================"
	@echo " GNR638 Assignment 2 - Makefile"
	@echo "============================================================"
	$(ECHO_EMPTY)
	@echo " FIRST RUN:"
	@echo "   make ready              Create venv + install deps (CUDA)"
	@echo "   make ready-cpu          Create venv + install deps (CPU)"
	@echo "   make setup              Create train/val split"
	$(ECHO_EMPTY)
	@echo " RUN SCENARIOS:"
	@echo "   make all                Run all 5 scenarios (default 3 models)"
	@echo "   make scenario1          Scenario 1: Linear Probe"
	@echo "   make scenario2          Scenario 2: Fine-Tuning Strategies"
	@echo "   make scenario3          Scenario 3: Few-Shot Learning"
	@echo "   make scenario4          Scenario 4: Corruption Robustness"
	@echo "   make scenario5          Scenario 5: Layer-Wise Probing"
	@echo "   make test               Evaluate saved models on test set"
	$(ECHO_EMPTY)
	@echo " OVERRIDES (append to any target):"
	@echo "   MODELS=\"resnet50 inception_v3\"     Select models"
	@echo "   MODELS=all                         All 5 models"
	@echo "   DATA_DIR=./my_data                 Dataset path"
	@echo "   EPOCHS=10                          Max epochs (full data)"
	@echo "   EPOCHS_FEW=5                       Max epochs (few-shot)"
	@echo "   SEED=123                           Random seed"
	@echo "   BATCH_SIZE=64                      Batch size"
	@echo "   VENV_DIR=./my_venv                 Custom venv path"
	$(ECHO_EMPTY)
	@echo " EXAMPLES:"
	@echo "   make ready"
	@echo "   make setup"
	@echo "   make all"
	@echo "   make scenario1 MODELS=resnet50 EPOCHS=5"
	@echo "   make all MODELS=all EPOCHS=10"
	@echo "   make clean"
	$(ECHO_EMPTY)

# Create venv if it doesn't exist
$(VENV_PYTHON):
	@echo "[venv] Creating virtual environment at $(VENV_DIR) ..."
	$(SYS_PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	@echo "[venv] Virtual environment ready."

# Install dependencies (GPU with CUDA 12.8)
ready: $(VENV_PYTHON)
	@echo "[ready] Installing PyTorch with CUDA 12.8 ..."
	$(VENV_PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	@echo "[ready] Installing other dependencies ..."
	$(VENV_PYTHON) -m pip install timm ptflops umap-learn scikit-learn seaborn matplotlib numpy
	$(ECHO_EMPTY)
	@echo "============================================================"
	@echo " Environment ready! GPU-enabled PyTorch installed."
	@echo " Venv: $(VENV_DIR)"
	@echo " Run 'make setup' to create the train/val split."
	@echo " Run 'make all' to run all experiments."
	@echo "============================================================"

# Install dependencies (CPU only)
ready-cpu: $(VENV_PYTHON)
	@echo "[ready] Installing PyTorch (CPU only) ..."
	$(VENV_PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "[ready] Installing other dependencies ..."
	$(VENV_PYTHON) -m pip install timm ptflops umap-learn scikit-learn seaborn matplotlib numpy
	$(ECHO_EMPTY)
	@echo "============================================================"
	@echo " Environment ready! CPU-only PyTorch installed."
	@echo " Venv: $(VENV_DIR)"
	@echo "============================================================"

# Quick check
check: $(VENV_PYTHON)
	@echo "[check] Verifying installation ..."
	$(VENV_PYTHON) -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); import timm; print('timm:', timm.__version__)"

# Dataset setup
setup: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --setup --data-dir $(DATA_DIR) --results-dir $(RESULTS_DIR) --seed $(SEED)

# Individual scenarios
scenario1: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario 1 $(FLAGS)

scenario2: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario 2 $(FLAGS)

scenario3: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario 3 $(FLAGS)

scenario4: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario 4 $(FLAGS)

scenario5: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario 5 $(FLAGS)

# Run everything
all: $(VENV_PYTHON)
	$(VENV_PYTHON) main.py --scenario all $(FLAGS)

# Testing pipeline
test: $(VENV_PYTHON)
	@echo "[test] Running evaluation on test set from $(TEST_DIR) ..."
	@$(VENV_PYTHON) -c "import os, subprocess; models = '$(MODELS)'.split(); [subprocess.call(['$(VENV_PYTHON)', 'test.py', '--model', m, '--weights', '$(RESULTS_DIR)/scenario2/' + m + '_full_finetune.pth', '--test-dir', '$(TEST_DIR)', '--batch-size', '$(BATCH_SIZE)']) for m in models if os.path.exists('$(RESULTS_DIR)/scenario2/' + m + '_full_finetune.pth')]"

# Cleanup
clean:
	@echo "Removing results directory ..."
ifeq ($(OS),Windows_NT)
	@if exist "$(RESULTS_DIR)" rmdir /s /q "$(RESULTS_DIR)"
else
	rm -rf $(RESULTS_DIR)
endif
	@echo "Done."
