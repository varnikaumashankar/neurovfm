"""
Medical Image Loading and Preprocessing

load_image: Load any medical image (NIfTI/DICOM/NumPy)
load_numpy_as_sitk: Load NumPy array as SimpleITK image
prepare_for_inference: Convert to model-ready arrays
DatasetMetadata: Manage dataset metadata
CacheManager: Build preprocessing cache
"""

from .io import load_image, load_numpy_as_sitk
from .preprocess import prepare_for_inference
from .metadata import DatasetMetadata
from .cache import CacheManager
from .utils import crop_to_patch_divisible

__all__ = ['load_image', 'load_numpy_as_sitk', 'prepare_for_inference',
           'DatasetMetadata', 'CacheManager', 'crop_to_patch_divisible']