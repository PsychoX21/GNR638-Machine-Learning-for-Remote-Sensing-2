#!/bin/bash
set -e

echo "============================================"
echo "  DL MCQ Solver — Environment Setup"
echo "============================================"

WORK_DIR="$(pwd)"

# 1. Clone repository and copy project files
echo "[1/4] Cloning repository..."
git clone --depth 1 https://github.com/PsychoX21/GNR638-Machine-Learning-for-Remote-Sensing-2.git _repo_temp

# Copy project files to current dir (skip setup.bash to avoid overwriting ourselves)
if [ -d "_repo_temp/Project" ]; then
    cd _repo_temp/Project
    for item in *; do
        if [ "$item" != "setup.bash" ]; then
            cp -r "$item" "$WORK_DIR/" 2>/dev/null || true
        fi
    done
    cd "$WORK_DIR"
elif [ -d "_repo_temp/project" ]; then
    cd _repo_temp/project
    for item in *; do
        if [ "$item" != "setup.bash" ]; then
            cp -r "$item" "$WORK_DIR/" 2>/dev/null || true
        fi
    done
    cd "$WORK_DIR"
fi
rm -rf _repo_temp

if [ ! -f "inference.py" ]; then
    echo "ERROR: inference.py not found after cloning."
    exit 1
fi
echo "  -> Files copied to ${WORK_DIR}"

# 2. Create conda environment
echo "[2/4] Creating conda environment..."
conda create -n gnr_project_env python=3.11 -y

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate gnr_project_env

# Install uv for fast, reliable dependency resolution
pip install uv==0.7.8

# Install vLLM with CUDA 12.6 backend (this pulls the correct torch automatically)
# vLLM 0.19.1 = last release before 0.20.0 switched to CUDA 13.0
uv pip install vllm==0.19.1 --torch-backend=cu126

# Install remaining pinned dependencies
uv pip install transformers==4.51.3 accelerate==1.6.0 huggingface_hub==0.30.2
uv pip install Pillow==11.2.1 numpy==2.2.5 pandas==2.2.3 sympy==1.13.3 openai==1.82.0 uvloop==0.21.0

# 4. Download model weights
echo "[4/4] Downloading Qwen/Qwen3.5-27B-FP8 weights..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3.5-27B-FP8',
    local_dir='${WORK_DIR}/weights/qwen35_27b_fp8',
    resume_download=True
)
print('  -> Download complete.')
"

# Verify
python -c "
import torch, transformers, vllm
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'vLLM: {vllm.__version__}')
print(f'Transformers: {transformers.__version__}')
"

echo "============================================"
echo "  Setup complete!"
echo "============================================"
