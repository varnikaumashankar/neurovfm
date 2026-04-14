#!/bin/bash
#SBATCH --job-name=neurovfm_age_regress
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w26_class
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chyhsu/Documents/logs/neurovfm_age_regress_%j.log
#SBATCH --error=/home/chyhsu/Documents/logs/neurovfm_age_regress_%j.err

set -e

mkdir -p /home/chyhsu/Documents/logs

echo "========================================================"
echo "NeuroVFM Age Regression Training Job"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${HOSTNAME}"
echo "Start:  $(date)"
echo "========================================================"

REPO_DIR="/home/chyhsu/Documents/neurovfm"
VENV_DIR="/home/chyhsu/Documents/neurovfm_venv_glibc28"

LABELS_PATH="/home/chyhsu/Documents/val_labels/participants_merged.tsv"
OUTPUT_DIR="/home/chyhsu/Documents/regression_runs/age_avgpool"
RUN_LOG_DIR="$OUTPUT_DIR/logs"

SUBJECT_COL="participant_id"
TARGET_COL="age"
STRIP_SUBJECT_PREFIX="sub-"

# Change this to "abmil" or "addmil" if you want to try the repo's MIL heads.
POOLER="avgpool"
HIDDEN_DIMS="256,64"
MIL_HIDDEN_DIM="256"

NUM_WORKERS="0"
BATCH_SIZE="8"
EPOCHS="50"
LR="1e-3"
WEIGHT_DECAY="1e-4"
VAL_FRACTION="0.2"
SEED="42"

RCLONE_REMOTE="chyhsu"
RCLONE_BASE_PATH="neurovfm"
REMOTE_EMBEDDING_SUBDIR="train_embeddings"
LOCAL_CACHE_DIR="/home/chyhsu/Documents/training_embedding_cache_age"
BATCH_START="1"
BATCH_END="33"
REMOTE_CHUNK_SIZE="100"

echo "Loading required modules..."
module load cuda/11.8.0
module load python/3.10.4
module load gcc/10.3.0
echo "Modules loaded successfully."

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run: sbatch scripts/setup_venv.sh first to create the environment"
    exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is not installed or not in PATH."
    exit 1
fi

echo "Activating pre-built virtual environment at $VENV_DIR..."
source "$VENV_DIR/bin/activate"
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

echo "Verifying environment..."
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "========================================================"
echo "Training Configuration"
echo "  Labels:          $LABELS_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Target:          $TARGET_COL"
echo "  Pooler:          $POOLER"
echo "  Hidden dims:     $HIDDEN_DIMS"
echo "  Epochs:          $EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  LR:              $LR"
echo "  Weight decay:    $WEIGHT_DECAY"
echo "  Val fraction:    $VAL_FRACTION"
echo "  Seed:            $SEED"
echo "  Batches:         $BATCH_START -> $BATCH_END"
echo "  Remote subdir:   $REMOTE_EMBEDDING_SUBDIR"
echo "  Local cache:     $LOCAL_CACHE_DIR"
echo "========================================================"

mkdir -p "$OUTPUT_DIR" "$RUN_LOG_DIR"

CMD=(
    python "$REPO_DIR/scripts/regression_head.py"
    --labels_path "$LABELS_PATH"
    --output_dir "$OUTPUT_DIR"
    --subject_col "$SUBJECT_COL"
    --target_col "$TARGET_COL"
    --strip_subject_prefix "$STRIP_SUBJECT_PREFIX"
    --pooler "$POOLER"
    --hidden_dims "$HIDDEN_DIMS"
    --mil_hidden_dim "$MIL_HIDDEN_DIM"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --val_fraction "$VAL_FRACTION"
    --seed "$SEED"
    --use_remote_batches
    --rclone_remote "$RCLONE_REMOTE"
    --rclone_base_path "$RCLONE_BASE_PATH"
    --remote_embedding_subdir "$REMOTE_EMBEDDING_SUBDIR"
    --local_cache_dir "$LOCAL_CACHE_DIR"
    --batch_start "$BATCH_START"
    --batch_end "$BATCH_END"
    --remote_chunk_size "$REMOTE_CHUNK_SIZE"
)

echo "--------------------------------------------------------"
echo "Running:"
printf ' %q' "${CMD[@]}"
echo
echo "--------------------------------------------------------"

"${CMD[@]}"

echo "========================================================"
echo "Training job completed successfully."
echo "End: $(date)"
echo "========================================================"
