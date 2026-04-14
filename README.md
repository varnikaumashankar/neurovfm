# NeuroVFM

## Health system learning achieves generalist neuroimaging models

[**Preprint**](https://arxiv.org/abs/2511.18640) / [**Interactive Demo**](https://neurovfm.mlins.org) / [**Models**](https://huggingface.co/collections/mlinslab/neurovfm) / [**MLiNS Lab**](https://mlins.org)

**NeuroVFM** is a health system–scale, volumetric foundation model for multimodal neuroimaging, trained with self-supervision on **5.24M** MRI/CT volumes (**567k** studies) spanning **20+ years** of routine clinical care at Michigan Medicine. 

![NeuroVFM overview](figures/MainFig1.png)

The NeuroVFM stack includes:

- **3D ViT encoder**, general-purpose representations for *any* clinical neuroimage (T1, T2, FLAIR, DWI, CT, etc.)
- **Study-level diagnostic heads**, covering **74 MRI**/**82 CT** expert-defined diagnoses for *any* neuroimaging study
- **Findings LLM**, generates preliminary findings given *any* neuroimaging study plus clinical context
- **Reasoning API**, pass outputs to a frontier reasoning model for higher-level tasks (e.g., triage)

> **Research use only.** Not a medical device. Do not use for clinical decision-making.

## 🔎 TL;DR (what NeuroVFM gives you)

NeuroVFM's defining feature is a standalone `pipelines/` package, which processes raw NIfTI/DICOM files given a study directory and returns (1) diagnostic probabilities, (2) findings, and (3) interpretation from a frontier reasoning model. All NeuroVFM models are hosted on HuggingFace; please request access [here](https://huggingface.co/collections/mlinslab/neurovfm).

```python
from neurovfm.pipelines import load_encoder, load_diagnostic_head, load_vlm, interpret_findings

# Load pretrained models from HuggingFace
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct")

# Load and preprocess a study directory with 1+ DICOM/NIfTI files
batch = preprocessor.load_study("/path/to/ct/study/", modality="ct")

# Generate embeddings and predictions
embeddings = encoder.embed(batch)
predictions = dx_head.predict(embeddings, batch)

# Load findings LLM
generator, preproc = load_vlm("mlinslab/neurovfm-llm")
vols = preproc.load_study("/path/to/study/")

# clinical_context = "LOC and nausea."                  # optional clinical context
clinical_context = None

findings = generator.generate(vols, clinical_context)

# optional: pass findings to external frontier LLM to interpret (e.g. clinical triage)
api_key = "..." # requires API key (e.g., OpenAI) set in your environment
intepretation = interpret_findings(findings, clinical_context, api_key)
```

## Installation

NeuroVFM is a standard Python package built on PyTorch (compiled with CUDA 12.4). To install it, clone this repository and install with `pip` (editable or regular). For efficient 3D ViT training and inference, NeuroVFM expects **FlashAttention-2 v2.6.3** built from source (including the fused dense/MLP and DropAddNorm kernels). FlashAttention-2 only supports recent NVIDIA GPUs with Tensor Cores; see the `flash-attn` README for exact GPU, CUDA, and PyTorch compatibility.

## This branch: regression head notebook workflow

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

