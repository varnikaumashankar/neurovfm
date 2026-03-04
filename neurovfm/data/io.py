"""
Medical Image I/O Module

This module provides a unified interface for loading and preprocessing medical images
from various formats (NIfTI, DICOM, NumPy) with automatic format detection.

The main entry point is the `load_image()` function, which automatically detects
the image format based on file extension or directory structure, loads the image,
and applies standardized preprocessing suitable for model inference.

Example Usage:
    >>> from neurovfm.io import load_image
    >>> 
    >>> # Load NIfTI file
    >>> img = load_image('/path/to/scan.nii.gz')
    >>> 
    >>> # Load DICOM series from directory
    >>> img = load_image('/path/to/dicom_series/')
    >>> 
    >>> # Load single DICOM file
    >>> img = load_image('/path/to/scan.dcm')
    >>> 
    >>> # Convert to numpy array
    >>> if img is not None:
    ...     import SimpleITK as sitk
    ...     arr = sitk.GetArrayFromImage(img)
    ...     print(f"Shape: {arr.shape}")
    >>> # Load NumPy array (e.g., OpenBHB Quasi-Raw)
    >>> img = load_image('sub-XXX_preproc-quasiraw_T1w.npy')
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import SimpleITK as sitk

from .utils import load_nifti_file, load_dicom_file, preprocess_image


def load_numpy_as_sitk(
    path,
    assumed_spacing=(1.0, 1.0, 1.0),
    assumed_z_dim=2,
):
    """
    Load a NumPy .npy file and wrap it as a SimpleITK Image.

    This is useful for datasets that distribute pre-processed volumes as NumPy
    arrays (e.g., OpenBHB Quasi-Raw T1w). Since .npy files lack spatial metadata
    (orientation, spacing, origin), reasonable defaults are assumed.

    Args:
        path (str or Path): Path to .npy file containing a 3D array [D, H, W].
        assumed_spacing (tuple): Physical spacing in mm to assign to the image.
            Default (1.0, 1.0, 1.0) is appropriate for MNI152-registered data.
        assumed_z_dim (int): Which axis (0, 1, or 2) is the slice/through-plane
            dimension. Default 2 (last axis = axial slices).

    Returns:
        SimpleITK.Image or None: 3D image with assigned spacing metadata,
            or None if loading fails or array is not 3D.
    """
    try:
        arr = np.load(str(path))
    except Exception as e:
        print(f"Error loading NumPy file {path}: {e}")
        return None

    # Squeeze leading singleton dims (some datasets save as (1, D, H, W))
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        print(f"Warning: Expected 3D array, got {arr.ndim}D shape {arr.shape}. Skipping {path}")
        return None

    # SimpleITK expects (x, y, z) ordering for spacing/size, but
    # GetImageFromArray interprets array as (z, y, x).
    img_sitk = sitk.GetImageFromArray(arr.astype(np.float64))
    img_sitk.SetSpacing(assumed_spacing)

    print(f"Loaded NumPy array: shape={arr.shape}, assumed spacing={assumed_spacing}")
    return img_sitk


def load_image(path, preprocess=True, assumed_spacing=(1.0, 1.0, 1.0), assumed_z_dim=2):
    """
    Load and preprocess a medical image with automatic format detection.
    
    This is the main entry point for loading medical images. It automatically
    detects whether the input is NIfTI, DICOM, or NumPy format and applies the
    appropriate loading method, followed by optional standardized preprocessing.
    
    Format Detection Logic:
    - Files ending in .nii or .nii.gz → NIfTI
    - Files ending in .dcm or .dicom → DICOM
    - Files ending in .npy → NumPy array (e.g., OpenBHB Quasi-Raw)
    - Directories → Assumed to contain DICOM series
    - Other extensions → Attempts generic SimpleITK loading (may fail)
    
    Preprocessing Pipeline (if preprocess=True):
    1. Reorients to standard RPI orientation
    2. Determines slice dimension based on spacing/size heuristics
    3. Resamples to anisotropic resolution (1x1x4mm)
    4. Crops to dimensions divisible by 16 (in-plane) and 4 (through-plane)
    
    For NumPy files, preprocessing crops dimensions to be patch-divisible but
    skips reorientation/resampling (since the data is already in a standard space).
    
    Args:
        path (str or Path): Path to medical image file or directory containing
            DICOM series. Supported formats:
            - NIfTI: .nii, .nii.gz
            - DICOM: .dcm, .dicom, or directory with DICOM series
            - NumPy: .npy (3D array)
        preprocess (bool, optional): Whether to apply standardized preprocessing
            (reorientation, resampling, cropping). Defaults to True.
        assumed_spacing (tuple, optional): Spacing to assign to NumPy arrays that
            lack spatial metadata. Default (1.0, 1.0, 1.0) for MNI152 data.
        assumed_z_dim (int, optional): Slice dimension for NumPy arrays. Default 2.
    
    Returns:
        SimpleITK.Image or None: Loaded (and optionally preprocessed) 3D image,
            or None if loading fails or image format is unsupported.
    
    Raises:
        FileNotFoundError: If the specified path doesn't exist.
    
    Examples:
        >>> # Load and preprocess a NIfTI file
        >>> img = load_image('brain_scan.nii.gz')
        >>> 
        >>> # Load DICOM without preprocessing
        >>> img = load_image('dicom_series/', preprocess=False)
        >>> 
        >>> # Load NumPy array (OpenBHB Quasi-Raw)
        >>> img = load_image('sub-XXX_preproc-quasiraw_T1w.npy', preprocess=True)
        >>> 
        >>> # Load and convert to numpy
        >>> img = load_image('scan.nii.gz')
        >>> if img is not None:
        ...     arr = sitk.GetArrayFromImage(img)
        ...     print(f"Loaded volume: {arr.shape}, dtype: {arr.dtype}")
    
    Notes:
        - Multi-component images (e.g., RGB) are not supported and will return None
        - DICOM series should be in a single directory with consistent orientation
        - Preprocessing uses BSpline interpolation for high-quality resampling
        - Target spacing: 1x1x4mm (in-plane × through-plane)
        - NumPy arrays are assumed to be already in standard orientation
    """
    path = Path(path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Determine format and load image
    img_sitk = None
    
    is_numpy = False
    
    if path.is_dir():
        # Directory: assume DICOM series
        print(f"Loading DICOM series from directory: {path}")
        img_sitk = load_dicom_file(path)
    else:
        # File: detect format by extension
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.gz']:
            # Handle .nii.gz
            if path.name.endswith('.nii.gz'):
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            elif suffix == '.nii':
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            else:
                # .gz but not .nii.gz, might be compressed NIfTI
                print(f"Loading file (assuming NIfTI): {path}")
                img_sitk = load_nifti_file(path)
        
        elif suffix in ['.dcm', '.dicom']:
            print(f"Loading DICOM file: {path}")
            img_sitk = load_dicom_file(path)
        
        elif suffix == '.npy':
            print(f"Loading NumPy file: {path}")
            img_sitk = load_numpy_as_sitk(path, assumed_spacing, assumed_z_dim)
            is_numpy = True
        
        else:
            # Unknown format: try generic loading
            print(f"Warning: Unknown format '{suffix}'. Attempting generic load: {path}")
            try:
                img_sitk = sitk.ReadImage(str(path))
                if img_sitk.GetNumberOfComponentsPerPixel() > 1:
                    print(f"Warning: Multi-component image detected. Skipping {path}")
                    return None
            except Exception as e:
                print(f"Error loading file {path}: {e}")
                return None
    
    # Check if loading was successful
    if img_sitk is None:
        print(f"Failed to load image from: {path}")
        return None
    
    # Apply preprocessing if requested
    if preprocess:
        if is_numpy:
            # NumPy arrays are already in standard space (e.g., MNI152).
            # Skip reorientation/resampling; only crop to patch-divisible dims.
            print(f"Applying NumPy-specific preprocessing (crop only)...")
            try:
                from .utils import crop_to_patch_divisible
                img_sitk = crop_to_patch_divisible(img_sitk, z_dim=assumed_z_dim)
            except Exception as e:
                print(f"Error during NumPy preprocessing: {e}")
                return None
        else:
            print(f"Applying standardized preprocessing...")
            try:
                img_sitk = preprocess_image(img_sitk)
            except Exception as e:
                print(f"Error during preprocessing: {e}")
                return None
    
    print(f"Successfully loaded image: shape={img_sitk.GetSize()}, spacing={img_sitk.GetSpacing()}")
    return img_sitk

