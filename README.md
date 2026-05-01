# NeuroVFM: clean-coordinate branch

This branch isolates the attention-to-coordinate workflow. It keeps the shared NeuroVFM code mostly unchanged and adds a focused notebook for mapping MIL attention scores back into image space.

## What is special here

- Adds `coordinate.ipynb`, a compact pipeline for pairing token coordinates with saved attention weights.
- Converts token-level attention into a 3D patch map, then expands that patch map into a voxel-level attention volume.
- Normalizes attention maps and overlays them on MRI slices for visual inspection.
- Includes subject ID normalization so files named like `sub-*_encoder_coords.pt` and `subject_*_attention.csv` can be merged reliably.

## Start from these files

- `coordinate.ipynb`: the branch-specific workflow.
- `neurovfm/pipelines/preprocessor.py`: produces token coordinates during preprocessing.
- `neurovfm/models/mil.py`: source of the MIL attention weights used by the notebook.

## Expected inputs

- Coordinate tensors saved as `sub-<id>_encoder_coords.pt`.
- Attention CSV files saved as `subject_<id>_attention.csv` with an `attention_weight` column.
- The corresponding MRI volume if you want to draw slice overlays.

## When to use this branch

Use `clean-coordinate` when you already have embeddings, token coordinates, and attention weights, and you want a clean path from model attention back to anatomical visualization.

## License

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 license on HuggingFace; request access with an institutional email.
