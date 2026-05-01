# NeuroVFM: neurovfm_baseline branch

This branch is focused on a baseline regression-head workflow on top of saved NeuroVFM encoder embeddings. It is more scriptable than the notebook-heavy regression branch and is set up for age prediction runs.

## What is special here

- Adds a reusable `scripts/regression_head.py` trainer for continuous targets such as age.
- Supports three pooling choices over variable-length token embeddings: average pooling, AB-MIL, and Add-MIL.
- Can read local embedding files or download remote embedding batches through `rclone` before training.
- Saves practical run artifacts: `best_model.pt`, `history.json`, `val_predictions.csv`, validation metrics, and optional attention summaries for MIL runs.
- Adds plotting and SLURM helpers for running and reviewing age regression experiments.

## Start from these files

- `scripts/regression_head.py`: command-line training entry point.
- `scripts/regression_head.ipynb`: notebook version of the same baseline workflow.
- `scripts/plot_regression_results.py`: generates loss, prediction, age distribution, error-by-bin, and validation metric plots.
- `scripts/submit_train_age_regression.sh`: Great Lakes job script for an age-regression run.
- `scripts/extract_embeddings.py`: upstream embedding extraction helper.

## Expected inputs

- A CSV or TSV label table with one row per subject.
- A subject ID column and a continuous target column, for example `age`.
- Embedding files named like `<subject_id>_encoder_embeddings.pt`.

## When to use this branch

Use `neurovfm_baseline` when you want a reproducible downstream baseline on frozen NeuroVFM embeddings, especially for age regression or another single continuous outcome.

## License

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 license on HuggingFace; request access with an institutional email.
