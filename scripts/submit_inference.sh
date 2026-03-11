#!/bin/bash
#SBATCH --job-name=neurovfm_extract
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w26_class
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chyhsu/Documents/logs/neurovfm_extract_%j.log
#SBATCH --error=/home/chyhsu/Documents/logs/neurovfm_extract_%j.err

set -e

echo "Loading required modules..."
module load cuda/11.8.0
module load python/3.10.4
module load gcc/10.3.0
echo "Modules loaded successfully."

VENV_DIR="$HOME/neurovfm_venv"

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

echo "Creating fresh virtual environment..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel packaging ninja
echo "✓ Build tools upgraded"

echo "Installing PyTorch 2.5.0 with CUDA 11.8..."
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"

echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
echo "✓ torch-scatter installed"

echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation
echo "✓ flash-attn installed"

echo "Installing PyTorch Lightning and metrics..."
pip install pytorch-lightning==2.5.0.post0 torchmetrics
echo "✓ PyTorch Lightning installed"

echo "Installing core dependencies..."
pip install omegaconf==2.3.0 nibabel==5.3.2 matplotlib==3.10.7 huggingface-hub==0.34.4 einops==0.8.0 SimpleITK==2.4.0 numpy==2.1.2 pandas==2.2.3 tqdm==4.66.5 scipy==1.15.2 scikit-learn==1.6.1 positional-encodings==6.0.3
echo "✓ Core dependencies installed"

echo "Installing ML framework dependencies..."
pip install outlines==1.1.1 transformers peft timm pydantic openai
echo "✓ ML dependencies installed"

echo "Installing neurovfm package..."
pip install -e . --no-deps
echo "✓ neurovfm installed"

echo "Verifying imports..."
python -c "import torch; import flash_attn; import einops; import torchmetrics; print('All packages verified')"
echo "✓ Verification complete"

INPUT_NPY_DIR="/home/chyhsu/Documents/data/validate"
NIFTI_DIR="/home/chyhsu/Documents/nii_gz"

export PYTHONPATH="$PWD:$PYTHONPATH"
MODALITY="mri"
BASE_OUT_DIR="/home/chyhsu/Documents/output"

echo "--------------------------------------------------------"
echo "Starting NeuroVFM Inference on $HOSTNAME"
echo "Target Modality: $MODALITY"
echo "Base Output Directory: $BASE_OUT_DIR"
echo "--------------------------------------------------------"

COUNT=0
MAX_FILES=5

for INPUT_FILE in "$NIFTI_DIR"/*.nii.gz; do
    if [ ! -f "$INPUT_FILE" ]; then
        echo "No .nii.gz files found in $NIFTI_DIR. Skipping."
        break
    fi
    if [ "$COUNT" -ge "$MAX_FILES" ]; then
        echo "Reached $MAX_FILES files. Stopping."
        break
    fi
    echo "Processing ($((COUNT+1))/$MAX_FILES): $INPUT_FILE"
    python scripts/extract_features.py --study_path "$INPUT_FILE" --modality "$MODALITY" --output_base_dir "$BASE_OUT_DIR"
    COUNT=$((COUNT+1))
    echo "Finished processing $INPUT_FILE"
    echo "--------------------------------------------------------"
done

echo "All jobs completed successfully."
