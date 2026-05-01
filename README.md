# NeuroVFM: npy branch

This branch adds direct support for 3D NumPy volumes, especially OpenBHB-style quasi-raw T1w `.npy` files. It is useful when the input data is already registered and should not go through the standard NIfTI/DICOM resampling path.

## What is special here

- Extends image loading so `load_image()` can detect `.npy` files and wrap them as SimpleITK images.
- Uses NumPy-specific preprocessing: assign assumed spacing, skip reorientation/resampling, and crop only to patch-divisible dimensions.
- Updates the study preprocessor so directories can contain `.npy` files alongside NIfTI or DICOM inputs.
- Adds an example for running the encoder directly on OpenBHB quasi-raw `.npy` inputs.
- Adds fallback implementations for FlashAttention fused dense and MLP components, plus an SDPA fallback for the language model when a CPU FlashAttention shim is detected.

## Start from these files

- `examples/inference_encoder_npy.py`: minimal direct `.npy` encoder example.
- `neurovfm/data/io.py`: `.npy` detection and SimpleITK wrapping.
- `neurovfm/data/utils.py`: `crop_to_patch_divisible()` for lightweight preprocessing.
- `neurovfm/pipelines/preprocessor.py`: includes `.npy` files in study directory discovery.
- `neurovfm/models/vit.py`, `neurovfm/models/patch_embed.py`, and `neurovfm/models/vlm.py`: attention and fused-layer compatibility changes.

## When to use this branch

Use `npy` when your volumes are already stored as 3D NumPy arrays in a standard space and converting them to NIfTI would add unnecessary preprocessing or bookkeeping.

## Important assumptions

`.npy` files do not carry orientation, origin, or spacing metadata. This branch assumes the arrays are already in a standard orientation and assigns default spacing before tokenization.

## License

Code is released under the MIT License. Model weights are provided under the CC-BY-NC-SA 4.0 license on HuggingFace; request access with an institutional email.
