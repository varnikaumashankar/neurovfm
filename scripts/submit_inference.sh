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

mkdir -p /home/chyhsu/Documents/logs

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

export HF_HUB_OFFLINE=1

echo "Verifying environment..."
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)"
python "$REPO_DIR/scripts/extract_embeddings.py" --help >/dev/null

BATCH_START="${BATCH_START:-6}"
BATCH_END="${BATCH_END:-10}"
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
    NIFTI_DIR="/home/chyhsu/Documents/nii_gz/batch_${BATCH_ID}"
    EMBEDDING_DIR="/home/chyhsu/Documents/output/batch_${BATCH_ID}"
    COORDS_DIR="/home/chyhsu/Documents/output_coords/batch_${BATCH_ID}"

    REMOTE_NIFTI_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/train_nii_gz/batch_${BATCH_ID}"
    REMOTE_EMBEDDING_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/train_embeddings/batch_${BATCH_ID}"
    REMOTE_COORDS_PATH="${RCLONE_REMOTE}:${RCLONE_BASE_PATH}/train_coords/batch_${BATCH_ID}"

    echo "========================================================"
    echo "Batch ${BATCH_ID}"
    echo "Input remote: ${REMOTE_NIFTI_PATH}"
    echo "Embedding remote: ${REMOTE_EMBEDDING_PATH}"
    echo "Coords remote: ${REMOTE_COORDS_PATH}"
    echo "========================================================"

    echo "Preparing local batch directories..."
    mkdir -p "$NIFTI_DIR" "$EMBEDDING_DIR" "$COORDS_DIR"

    echo "Downloading batch_${BATCH_ID} .nii.gz files from ${REMOTE_NIFTI_PATH} ..."
    rclone copy "$REMOTE_NIFTI_PATH" "$NIFTI_DIR" --include "*.nii.gz"

    echo "--------------------------------------------------------"
    echo "Starting NeuroVFM Embedding Extraction on $HOSTNAME"
    echo "Target Modality: $MODALITY"
    echo "Embedding Output Directory: $EMBEDDING_DIR"
    echo "Coords Output Directory: $COORDS_DIR"
    echo "--------------------------------------------------------"

    COUNT=0
    SKIP_COUNT=0
    MAX_FILES=

    for INPUT_FILE in "$NIFTI_DIR"/*.nii.gz; do
        if [ ! -f "$INPUT_FILE" ]; then
            echo "No .nii.gz files found in $NIFTI_DIR. Skipping."
            break
        fi
        INPUT_BASENAME="$(basename "$INPUT_FILE")"
        SUBJECT_ID="${INPUT_BASENAME%%_*}"
        if [[ "$SUBJECT_ID" != sub-* ]]; then
            SUBJECT_ID="${INPUT_BASENAME%.nii.gz}"
        fi
        EXPECTED_EMBEDDING_PATH="$EMBEDDING_DIR/${SUBJECT_ID}_encoder_embeddings.pt"
        EXPECTED_COORDS_PATH="$COORDS_DIR/${SUBJECT_ID}_encoder_coords.pt"
        if [ -f "$EXPECTED_EMBEDDING_PATH" ] && [ -f "$EXPECTED_COORDS_PATH" ]; then
            echo "Embedding and coords already exist for ${INPUT_BASENAME}. Skipping."
            SKIP_COUNT=$((SKIP_COUNT+1))
            continue
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
        python "$REPO_DIR/scripts/extract_embeddings.py" --study_path "$INPUT_FILE" --modality "$MODALITY" --output_base_dir "$EMBEDDING_DIR" --coords_output_dir "$COORDS_DIR"
        COUNT=$((COUNT+1))
        echo "Finished processing $INPUT_FILE"
        echo "--------------------------------------------------------"
    done

    if [ "$COUNT" -eq 0 ] && [ "$SKIP_COUNT" -gt 0 ]; then
        echo "Batch ${BATCH_ID} is already complete locally."
    else
        echo "Batch ${BATCH_ID}: generated ${COUNT} embedding files, skipped ${SKIP_COUNT} existing files."
    fi

    # echo "Ensuring remote output directories exist..."
    # rclone mkdir "$REMOTE_NIFTI_PATH"
    # rclone mkdir "$REMOTE_EMBEDDING_PATH"

    # echo "Uploading converted .nii.gz files to ${REMOTE_NIFTI_PATH} ..."
    # rclone copy "$NIFTI_DIR" "$REMOTE_NIFTI_PATH" --include "*.nii.gz"

    # echo "Uploading embedding .pt files to ${REMOTE_EMBEDDING_PATH} ..."
    # rclone copy "$EMBEDDING_DIR" "$REMOTE_EMBEDDING_PATH" --include "*.pt"
    echo "Uploading coords .pt files to ${REMOTE_COORDS_PATH} ..."
    rclone copy "$COORDS_DIR" "$REMOTE_COORDS_PATH" --include "*.pt"
    
    echo "Upload complete. Cleaning local batch files..."
    rm -rf "$NIFTI_DIR" "$EMBEDDING_DIR" "$COORDS_DIR"
done

echo "All embedding jobs completed successfully."
