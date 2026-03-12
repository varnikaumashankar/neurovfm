#!/bin/bash
#SBATCH --job-name=neurovfm_extract
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w26_class
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chyhsu/Documents/logs/neurovfm_extract_%j.log
#SBATCH --error=/home/chyhsu/Documents/logs/neurovfm_extract_%j.err

set -e

REPO_DIR="/home/chyhsu/Documents/neurovfm"

echo "Loading required modules..."
module load cuda/11.8.0
module load python/3.10.4
module load gcc/10.3.0
echo "Modules loaded successfully."

VENV_DIR="/home/chyhsu/Documents/neurovfm_venv_glibc28"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run: sbatch scripts/setup_venv.sh first to create the environment"
    exit 1
fi

echo "Activating pre-built virtual environment at $VENV_DIR..."
source "$VENV_DIR/bin/activate"
# if [ -f "$HOME/.bashrc" ]; then
#     source "$HOME/.bashrc"
# fi
echo "✓ Virtual environment activated"

echo "Python executable:"
which python
python -c "import sys; print(sys.executable)"

echo "FlashAttention check:"
python - <<'PY'
import sys
print("sys.path[0:5]=", sys.path[:5])

try:
    import flash_attn
    print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))
    print("flash_attn file:", flash_attn.__file__)
except Exception as e:
    print("flash_attn import failed:", repr(e))

try:
    import fused_dense_lib
    print("fused_dense_lib import OK:", fused_dense_lib)
except Exception as e:
    print("fused_dense_lib import failed:", repr(e))

try:
    from flash_attn.ops.fused_dense import FusedDense
    print("FusedDense import OK")
except Exception as e:
    print("FusedDense import failed:", repr(e))
PY

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

echo "Verifying environment..."
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)"
python "$REPO_DIR/scripts/extract_embeddings.py" --help >/dev/null

BATCH_START="${BATCH_START:-1}"
BATCH_END="${BATCH_END:-8}"
if [ -n "${BATCH_ID:-}" ]; then
    BATCH_START="$BATCH_ID"
    BATCH_END="$BATCH_ID"
fi
RCLONE_REMOTE="${RCLONE_REMOTE:-chyhsu}"
RCLONE_BASE_PATH="${RCLONE_BASE_PATH:-neurovfm}"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
MODALITY="mri"

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is not installed or not in PATH."
    exit 1
fi

for BATCH_ID in $(seq "$BATCH_START" "$BATCH_END"); do
    INPUT_NPY_DIR="/home/chyhsu/Documents/data/batch_${BATCH_ID}"
    NIFTI_DIR="/home/chyhsu/Documents/nii_gz/batch_${BATCH_ID}"
    BASE_OUT_DIR="/home/chyhsu/Documents/output/batch_${BATCH_ID}"
    EMBEDDING_DIR="$BASE_OUT_DIR"

    REMOTE_BATCH_NPY_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/batches/batch_${BATCH_ID}"
    REMOTE_NIFTI_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/nii_gz/batch_${BATCH_ID}"
    REMOTE_EMBEDDING_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/embeddings/batch_${BATCH_ID}"

    echo "========================================================"
    echo "Batch ${BATCH_ID}"
    echo "Input remote: ${REMOTE_BATCH_NPY_PATH}"
    echo "========================================================"

    echo "Preparing local batch directories..."
    mkdir -p "$INPUT_NPY_DIR" "$NIFTI_DIR" "$EMBEDDING_DIR"

    echo "Downloading batch_${BATCH_ID} .npy files from ${REMOTE_BATCH_NPY_PATH} ..."
    rclone copy "$REMOTE_BATCH_NPY_PATH" "$INPUT_NPY_DIR" --include "*.npy"

    echo "Converting .npy files to .nii.gz..."
    python "$REPO_DIR/scripts/convert_npy_to_nifti.py" --input_dir "$INPUT_NPY_DIR" --output_dir "$NIFTI_DIR"

    echo "--------------------------------------------------------"
    echo "Starting NeuroVFM Embedding Extraction on $HOSTNAME"
    echo "Target Modality: $MODALITY"
    echo "Base Output Directory: $BASE_OUT_DIR"
    echo "--------------------------------------------------------"

    COUNT=0
    MAX_FILES=

    for INPUT_FILE in "$NIFTI_DIR"/*.nii.gz; do
        if [ ! -f "$INPUT_FILE" ]; then
            echo "No .nii.gz files found in $NIFTI_DIR. Skipping."
            break
        fi
        if [ -n "$MAX_FILES" ] && [ "$COUNT" -ge "$MAX_FILES" ]; then
            echo "Reached $MAX_FILES files. Stopping."
            break
        fi
        if [ -n "$MAX_FILES" ]; then
            echo "Processing ($((COUNT+1))/$MAX_FILES): $INPUT_FILE"
        else
            echo "Processing ($((COUNT+1))): $INPUT_FILE"
        fi
        python "$REPO_DIR/scripts/extract_embeddings.py" --study_path "$INPUT_FILE" --modality "$MODALITY" --output_base_dir "$BASE_OUT_DIR"
        COUNT=$((COUNT+1))
        echo "Finished processing $INPUT_FILE"
        echo "--------------------------------------------------------"
    done

    echo "Ensuring remote output directories exist..."
    rclone mkdir "$REMOTE_NIFTI_PATH"
    rclone mkdir "$REMOTE_EMBEDDING_PATH"

    echo "Uploading converted .nii.gz files to ${REMOTE_NIFTI_PATH} ..."
    rclone copy "$NIFTI_DIR" "$REMOTE_NIFTI_PATH" --include "*.nii.gz"

    echo "Uploading embedding .pt files to ${REMOTE_EMBEDDING_PATH} ..."
    rclone copy "$EMBEDDING_DIR" "$REMOTE_EMBEDDING_PATH" --include "*.pt"

    echo "Upload complete. Cleaning local batch files..."
    rm -rf "$INPUT_NPY_DIR" "$NIFTI_DIR" "$BASE_OUT_DIR"
done

echo "All embedding jobs completed successfully."
