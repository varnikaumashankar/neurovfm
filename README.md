# NeuroVFM: model_with_regressionhead branch

This branch is for downstream prediction experiments on saved NeuroVFM embeddings. It adds regression, classification, and attention-analysis workflows without changing the shared package goal.

## What is special here

- Adds notebook-driven training for OpenBHB and ADNI embeddings, including regression, classification, joint objectives, diagnostics, and hyperparameter tuning.
- Extends the MIL experiment code so AB-MIL and Add-MIL style heads can return attention weights and patch-level details for later analysis.
- Includes experimental MLP variants for regression, classification, and dual-output heads, with configurable hidden layers, dropout, and initialization.
- Adds attention-coordinate utilities for turning token attention into patch maps, voxel-level maps, and slice overlays.
- Keeps root-level experiment files separate from the package modules so this branch can be used as an exploratory workspace.

## Start from these files

- `training.ipynb`: main notebook for training and evaluating MLP heads on saved embeddings.
- `abmil.ipynb`: AB-MIL exploration notebook.
- `neurovfm/models/abmil.ipynb` and `neurovfm/models/abmil_classification.ipynb`: packaged notebook variants for regression and classification.
- `mil.py`: experimental MIL heads with attention-return paths and multiple attention scoring options.
- `projector.py`: experimental MLP heads for regression, classification, and dual tasks.
- `coordinate.ipynb`: joins saved coordinates with attention outputs and builds attention overlays.
- `basic_cnn.py`: small 2D regression CNN baseline.

## When to use this branch

Use `model_with_regressionhead` when the main task is comparing downstream heads on fixed NeuroVFM embeddings, inspecting attention behavior, or prototyping age/regression and classification objectives in notebooks.

## Notes

The branch expects saved encoder embeddings as inputs for most workflows. It is not the cleanest branch for package-only inference; use `main` for that.

## License

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 license on HuggingFace; request access with an institutional email.
