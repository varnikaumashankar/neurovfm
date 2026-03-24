#!/bin/bash
#SBATCH --job-name=examine_embeddings
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w26_class
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chyhsu/Documents/logs/examine_embeddings_%j.log
#SBATCH --error=/home/chyhsu/Documents/logs/examine_embeddings_%j.err

set -euo pipefail

REPO_DIR="/home/chyhsu/Documents/neurovfm"
VENV_DIR="/home/chyhsu/Documents/neurovfm_venv_glibc28"
NOTEBOOK_PATH="$REPO_DIR/examine_embeddings.ipynb"
OUTPUT_NOTEBOOK="$REPO_DIR/examine_embeddings_executed.ipynb"

mkdir -p /home/chyhsu/Documents/logs

module purge
module load cuda/11.8.0
module load python/3.10.4
module load gcc/10.3.0

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLBACKEND=Agg

cd "$REPO_DIR"

echo "Running on host: $(hostname)"
echo "Python: $(which python)"
python -c "import torch; print('torch=', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())"

python -m nbconvert \
  --to notebook \
  --execute "$NOTEBOOK_PATH" \
  --ExecutePreprocessor.timeout=-1 \
  --output "$OUTPUT_NOTEBOOK"

echo "Notebook execution finished: $OUTPUT_NOTEBOOK"
