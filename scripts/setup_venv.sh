#!/bin/bash
#SBATCH --job-name=neurovfm_setup
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w26_class
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chyhsu/Documents/logs/neurovfm_setup_glibc28_%j.log
#SBATCH --error=/home/chyhsu/Documents/logs/neurovfm_setup_glibc28_%j.err

set -euo pipefail

echo "========================================"
echo "NeuroVFM environment setup started"
echo "Host: $(hostname)"
date
echo "========================================"

PROJECT_DIR="/home/chyhsu/Documents/neurovfm"
VENV_DIR="/home/chyhsu/Documents/neurovfm_venv_glibc28"
PYTHON_BIN="/sw/pkgs/arc/python/3.10.4/bin/python3"

PYTHON_MODULE="python/3.10.4"
CUDA_MODULE="cuda/11.8.0"
GCC_MODULE="gcc/10.3.0"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
TORCH_VERSION="2.5.0"
TORCHVISION_VERSION="0.20.0"
TORCHAUDIO_VERSION="2.5.0"
TORCH_SCATTER_WHEEL_URL="https://data.pyg.org/whl/torch-2.5.0+cu118.html"

TRY_FLASH_ATTN="false"
FLASH_ATTN_VERSION="2.5.8"

module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"
module load "$GCC_MODULE"

echo "Loaded modules:"
module list 2>&1 || true

if [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "Installing uv..."
    python -m pip install --user uv
fi

"$HOME/.local/bin/uv" --version

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

echo "Creating venv..."
"$HOME/.local/bin/uv" venv "$VENV_DIR" --python "$PYTHON_BIN" --seed
source "$VENV_DIR/bin/activate"

echo "Python after activation:"
which python
python -V
python -m pip -V

echo "Installing build helpers..."
"$HOME/.local/bin/uv" pip install -U pip setuptools wheel packaging ninja psutil

echo "Installing torch / torchvision / torchaudio..."
"$HOME/.local/bin/uv" pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"

echo "Checking torch and CUDA..."
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

nvcc --version || true
ldd --version || true

echo "Installing torch-scatter..."
"$HOME/.local/bin/uv" pip install torch-scatter -f "$TORCH_SCATTER_WHEEL_URL"

echo "Installing lightning + metrics..."
"$HOME/.local/bin/uv" pip install pytorch-lightning==2.5.0.post0 torchmetrics

echo "Installing core scientific deps..."
"$HOME/.local/bin/uv" pip install \
    omegaconf==2.3.0 \
    nibabel==5.3.2 \
    matplotlib==3.10.7 \
    huggingface-hub==0.34.4 \
    einops==0.8.0 \
    SimpleITK==2.4.0 \
    numpy==2.1.2 \
    pandas==2.2.3 \
    tqdm==4.66.5 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    positional-encodings==6.0.3

echo "Installing ML framework deps..."
"$HOME/.local/bin/uv" pip install \
    outlines==1.1.1 \
    transformers \
    peft \
    timm \
    pydantic \
    openai

echo "Installing neurovfm from local repo..."
cd "$PROJECT_DIR"
"$HOME/.local/bin/uv" pip install -e . --no-deps

echo "Verifying base imports..."
python - <<'PY'
import torch
import einops
import torchmetrics
import nibabel
import transformers
print("Base environment verified")
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
PY

if [ "$TRY_FLASH_ATTN" = "true" ]; then
    echo "Trying flash-attn as best-effort only..."
    export MAX_JOBS=1
    export NVCC_THREADS=1

    python -m pip uninstall -y flash-attn || true
    rm -rf ~/.cache/uv ~/.cache/pip /tmp/pip-install-* /tmp/pip-ephem-wheel-cache-* || true

    set +e
    python -m pip install \
        --no-build-isolation \
        --no-binary :all: \
        --no-cache-dir \
        "flash-attn==$FLASH_ATTN_VERSION"

    python - <<'PY'
try:
    import flash_attn
    print("flash_attn import ok")
except Exception as e:
    print("flash_attn import failed:", repr(e))
PY
    set -e
else
    echo "Skipping flash-attn install on this cluster."
fi

echo "========================================"
echo "Setup completed"
echo "Venv location: $VENV_DIR"
echo "Note: neurovfm import may still fail unless flash-attn is available or the code is patched to use a fallback."
date
echo "========================================"