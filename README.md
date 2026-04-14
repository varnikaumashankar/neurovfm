# This branch: regression head notebook workflow

This branch is focused on a simple notebook workflow for training a regression head on top of saved NeuroVFM embeddings.

The main file to look at in this branch is:

- `scripts/regression_head.ipynb`

This notebook is designed for users who want to work step by step in Jupyter instead of running the full training script from the command line.

### What `regression_head.ipynb` does

The notebook trains a small model to predict a continuous target, such as age, from precomputed NeuroVFM encoder embeddings saved on disk.

It assumes you already have:

- a labels table with one row per subject
- a subject ID column
- a target column such as age
- a directory of embedding files ending in `_encoder_embeddings.pt`

### Notebook structure

The notebook is organized as a short local workflow:

1. **Introduction and setup guide**
   - Explains the purpose of the notebook, what files you need, and which settings to edit.

2. **Imports and shared definitions**
   - Loads the Python packages and helper functions used for reading labels, loading embeddings, building datasets, and training the regression model.

3. **Configuration cell**
   - Lets you choose the labels file, embedding directory, output directory, target column, pooling method, optimization settings, and device.

4. **Quick data check**
   - Verifies how many embedding files were found, how many label rows are valid, and how many subjects match between the two.

5. **Training cell**
   - Splits the matched data into train and validation sets, computes optional feature normalization, trains the regression head, and saves outputs such as `best_model.pt`, `history.json`, and `val_predictions.csv`.

6. **Plotting section**
   - Generates summary plots from a saved run directory, including the loss curve and validation prediction plots.

### How this connects to NeuroVFM

This branch does **not** retrain the NeuroVFM encoder itself. Instead, it uses NeuroVFM as a frozen feature extractor upstream and trains a lightweight downstream regression model on the saved embeddings.

In practice, the connection is:

- NeuroVFM encoder produces study embeddings
- embeddings are saved as `.pt` files on disk
- `scripts/regression_head.ipynb` loads those embeddings locally
- the notebook trains a regression head using components from `neurovfm.models`

The notebook uses the NeuroVFM model building blocks already provided in this repository, including:

- `MLP`
- `AggregateThenClassify` (AB-MIL)
- `ClassifyThenAggregate` (Add-MIL)

This means the branch is useful if you want to evaluate or fine-tune a simple prediction head on top of fixed NeuroVFM representations without modifying the main encoder training pipeline.

## Training

Code to reproduce the main experiments in our manuscript are provided under `training/`. We provide a cached dataset feature that allows users to initially preprocess their data using our pipeline and save them as `.pt` files for faster subsequent training. For more information, see `training/README.md.`

## LICENSE

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 LICENSE on HuggingFace; please request access with your institutional email.

