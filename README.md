# NeuroVFM: main branch

This is the reference branch for the shared NeuroVFM package and inference stack. It is the branch to use when you want the maintained package layout, HuggingFace-backed pipeline entry points, and the current utility scripts.

## What is special here

- Keeps the standard `neurovfm.pipelines` API for loading the encoder, diagnostic heads, findings generator, and reasoning helper.
- Includes script-based workflows for Great Lakes or local runs, including embedding extraction, feature extraction, inference submission, virtualenv setup, and `.npy` to NIfTI conversion.
- Restores the embedding inspection notebooks alongside the package tests, so this branch is a clean baseline for comparing the experiment branches.

## Start from these files

- `neurovfm/pipelines/`: load pretrained encoder, diagnostic, generator, and interpretation components.
- `scripts/extract_embeddings.py`: extract and save encoder embeddings for one study.
- `scripts/extract_features.py`: feature extraction workflow.
- `scripts/convert_npy_to_nifti.py`: convert NumPy volumes to `.nii.gz` before using the standard preprocessor.
- `scripts/submit_inference.sh` and `scripts/submit_examine_embeddings.sh`: SLURM entry points for batch jobs.
- `examples/`: minimal package usage examples and configuration files.

## When to use this branch

Use `main` when you need the common NeuroVFM codebase without a downstream regression, attention-coordinate, or direct `.npy` ingestion experiment layered on top.

## License

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 license on HuggingFace; request access with an institutional email.
