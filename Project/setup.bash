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

pip install "vllm>=0.11.0"
pip install "torch>=2.4.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.45.0" accelerate huggingface_hub
pip install Pillow numpy pandas sympy openai uvloop

# 4. Download model weights
echo "[4/4] Downloading Qwen3.5-27B-FP8 weights..."
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
"

echo "============================================"
echo "  Setup complete!"
echo "============================================"
