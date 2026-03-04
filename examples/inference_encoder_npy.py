"""
Example: Run NeuroVFM encoder on OpenBHB Quasi-Raw .npy files

OpenBHB distributes T1-weighted brain MRI as NumPy arrays (.npy).
The "Quasi-Raw" files (*_preproc-quasiraw_T1w.npy) are minimally
preprocessed and registered to MNI152 space at ~1mm isotropic.

This example shows how to feed them directly into the NeuroVFM pipeline.
"""

import torch
from neurovfm.pipelines import load_encoder

# 1. Load encoder + preprocessor from HuggingFace
encoder, preproc = load_encoder(
    "mlinslab/neurovfm-encoder",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 2. Load a single .npy file (OpenBHB Quasi-Raw T1w)
#    The pipeline auto-detects .npy format and applies lightweight
#    preprocessing (crop to patch-divisible dims only — no resampling
#    since the data is already in MNI152 standard space).
batch = preproc.load_study(
    "/path/to/sub-XXXX_preproc-quasiraw_T1w.npy",
    modality="mri",
)

# 3. Get token-level representations
embs = encoder.embed(batch)  # shape: [N_tokens, D]
print(f"Embeddings shape: {embs.shape}")

# --- OR: load a directory containing multiple .npy files ---
# batch = preproc.load_study("/path/to/openbhb_subjects/", modality="mri")
# embs = encoder.embed(batch)
