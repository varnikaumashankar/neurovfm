ROOT_DIR=/home/chyhsu/Documents # TODO: change this to your root directory
sbatch \
  --output="$ROOT_DIR/logs/neurovfm_setup_%j.log" \
  --error="$ROOT_DIR/logs/neurovfm_setup_%j.err" \
  $ROOT_DIR/neurovfm/scripts/setup_venv.sh