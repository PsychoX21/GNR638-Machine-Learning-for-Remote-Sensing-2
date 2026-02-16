# DeepNet Framework Makefile
# Cross-platform build system for Windows, Linux, and macOS

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    VENV_BIN := venv/Scripts
    VENV_ACTIVATE := venv\\Scripts\\activate
    MKDIR := mkdir
    RM := rmdir /s /q
    RM_FILE := del /f /q
    PATH_SEP := \\
    # Find the .pyd file in build directory (may be in Release/ subfolder on VS)
    # Copy to both project root and deepnet/ so imports work from any sys.path
    COPY_PYD = (if exist build\\Release\\deepnet_backend*.pyd (copy build\\Release\\deepnet_backend*.pyd . >nul & copy build\\Release\\deepnet_backend*.pyd deepnet\\ >nul) else (copy build\\deepnet_backend*.pyd . >nul & copy build\\deepnet_backend*.pyd deepnet\\ >nul))
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    endif
    PYTHON := python3
    VENV_BIN := venv/bin
    VENV_ACTIVATE := source venv/bin/activate
    MKDIR := mkdir -p
    RM := rm -rf
    RM_FILE := rm -f
    PATH_SEP := /
    COPY_PYD = cp build/deepnet_backend*.so . 2>/dev/null; cp build/deepnet_backend*.so deepnet/ 2>/dev/null || true
endif

# Configuration
BUILD_DIR := build
DATA ?= datasets/data_1
CONFIG ?= configs/mnist.yaml
MODEL ?= checkpoints/best_$(DATA).pth
EPOCHS ?= 50
VAL_SPLIT ?= 1.0

.PHONY: all setup build install clean distclean train eval test test-cuda test-layers test-gradient help

# Default target
all: setup build install

# Help
help:
	@echo "DeepNet Framework - Build System"
	@echo "================================="
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup       - Set up environment (first time only)"
	@echo "  2. Activate venv    - Run: $(VENV_ACTIVATE)"
	@echo "  3. make build       - Compile C++ backend"
	@echo "  4. make install     - Install Python package"
	@echo ""
	@echo "Or run all at once:"
	@echo "  make                - Does setup + build + install"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train          - Train model (use DATA=, CONFIG=, EPOCHS=)"
	@echo "  make eval           - Evaluate model (use DATA=, MODEL=)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests (layers, gradient, CUDA)"
	@echo "  make test-layers    - Run layer tests only"
	@echo "  make test-gradient  - Run gradient tests only"
	@echo "  make test-cuda      - Run CUDA GPU tests only"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make distclean      - Deep clean (also removes venv, pybind11)"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATA=datasets/data_1 EPOCHS=50"
	@echo "  make eval DATA=datasets/data_1 MODEL=checkpoints/best.pth"

# Setup virtual environment
setup:
	@echo "Setting up Python environment..."
	$(PYTHON) -m venv venv
	@echo "Installing dependencies..."
ifeq ($(DETECTED_OS),Windows)
	$(PYTHON) -m pip install --upgrade pip setuptools || echo "Pip upgrade skipped"
	$(VENV_BIN)\pip install -r requirements.txt
else
	$(VENV_BIN)/pip install --upgrade pip setuptools
	$(VENV_BIN)/pip install -r requirements.txt
endif
	@echo "Cloning pybind11..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist pybind11 git clone https://github.com/pybind/pybind11.git
else
	@if [ ! -d "pybind11" ]; then \
		git clone https://github.com/pybind/pybind11.git; \
	fi
endif
	@echo ""
	@echo "Setup complete!"
	@echo "Next step: Activate virtual environment with:"
	@echo "  $(VENV_ACTIVATE)"
	@echo "Then run: make build install"

# Build C++ backend
build:
	@echo "Building C++ backend (OS: $(DETECTED_OS))..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist $(BUILD_DIR) $(MKDIR) $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release -j8
else
	@$(MKDIR) $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release -j8
endif
	@echo "Build complete!"

# Install Python package
install:
	@echo "Installing Python package..."
ifeq ($(DETECTED_OS),Windows)
	@$(COPY_PYD)
else
	@$(COPY_PYD)
endif
	$(PYTHON) -m pip install -e .
	@echo "Installation complete!"
	@echo ""
	@echo "Verify with: make test"

# Train model
train:
	@echo "Training model on $(DATA)..."
	$(PYTHON) scripts/train.py \
		--dataset $(DATA) \
		--config $(CONFIG) \
		--epochs $(EPOCHS)

eval:
	@echo "Evaluating model on $(DATA) (split: $(VAL_SPLIT))..."
	$(PYTHON) scripts/evaluate.py \
		--dataset $(DATA) \
		--checkpoint $(MODEL) \
		--val-split $(VAL_SPLIT)


# Run all tests
test: test-layers test-gradient test-cuda test-cuda-ops test-device-ops test-gpu-integrity test-determinism
	@echo ""
	@echo "=== All tests completed ==="

# Individual test targets
test-layers:
	@echo "Running layer tests..."
	$(PYTHON) scripts/tests/test_all_layers.py

test-gradient:
	@echo "Running gradient tests..."
	$(PYTHON) scripts/tests/test_gradient.py

test-cuda:
	@echo "Running CUDA tests..."
	$(PYTHON) scripts/tests/test_cuda.py

test-cuda-ops:
	@echo "Running CUDA Ops tests..."
	$(PYTHON) scripts/tests/test_cuda_ops.py

test-device-ops:
	@echo "Running Device Ops verification..."
	$(PYTHON) scripts/tests/test_device_ops.py

test-gpu-integrity:
	@echo "Running GPU Integrity tests..."
	$(PYTHON) scripts/tests/test_gpu_integrity.py

test-determinism:
	@echo "Running Determinism tests..."
	$(PYTHON) scripts/tests/test_determinism.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(DETECTED_OS),Windows)
	@if exist $(BUILD_DIR) $(RM) $(BUILD_DIR)
	@if exist deepnet_backend*.pyd $(RM_FILE) deepnet_backend*.pyd
	@if exist deepnet\deepnet_backend*.pyd $(RM_FILE) deepnet\deepnet_backend*.pyd
	@if exist deepnet.egg-info $(RM) deepnet.egg-info
	@for /d %%i in (__pycache__) do @if exist %%i $(RM) %%i
else
	$(RM) $(BUILD_DIR)
	$(RM_FILE) deepnet_backend*.so deepnet_backend*.pyd
	$(RM_FILE) deepnet/deepnet_backend*.so deepnet/deepnet_backend*.pyd
	$(RM) deepnet.egg-info
	$(RM) __pycache__ deepnet/__pycache__ scripts/__pycache__ deepnet/python/__pycache__
	$(RM) .pytest_cache
endif
	@echo "Clean complete!"

# Deep clean (including venv and pybind11)
distclean: clean
	@echo "Deep cleaning (removing venv, pybind11, checkpoints, logs)..."
ifeq ($(DETECTED_OS),Windows)
	@if exist venv $(RM) venv
	@if exist pybind11 $(RM) pybind11
	@if exist checkpoints $(RM) checkpoints
	@if exist logs $(RM) logs
else
	$(RM) venv pybind11 checkpoints logs
endif
	@echo "Deep clean complete!"
