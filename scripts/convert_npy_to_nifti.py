import argparse
import os
import glob
import numpy as np
import logging
import itertools

try:
    import SimpleITK as sitk
except ImportError:
    print("Error: SimpleITK not installed. Run: pip install SimpleITK")
    import sys
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to .nii.gz format")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)
    
    npy_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not npy_files:
        logging.warning(f"No .npy files found in {args.input_dir}")
        return

    logging.info(f"Found {len(npy_files)} .npy files. Starting conversion...")

    for npy_path in npy_files:
        filename   = os.path.basename(npy_path)
        base_name  = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.nii.gz")

        try:
            arr = np.load(npy_path)
            
            # Log actual shape for debugging
            logging.info(f"{filename}: raw shape={arr.shape}, dtype={arr.dtype}")

            # Squeeze out any singleton batch/channel dimensions (e.g. [1,1,D,H,W] -> [D,H,W])
            arr = arr.squeeze()
            logging.info(f"{filename}: squeezed shape={arr.shape}")

            img = sitk.GetImageFromArray(arr)

            ndim = img.GetDimension()
            # Set 1mm isotropic spacing for all dimensions
            img.SetSpacing(tuple([1.0] * ndim))
            # Set standard MNI origin for first 3 dims, 0 for the rest
            mni_origin = (-90.0, -126.0, -72.0)
            img.SetOrigin(mni_origin[:ndim] if ndim <= 3 else mni_origin + (0.0,) * (ndim - 3))
            # Identity direction matrix (ndim x ndim flattened)
            identity = [1.0 if i == j else 0.0 for i, j in itertools.product(range(ndim), range(ndim))]
            img.SetDirection(identity)

            sitk.WriteImage(img, output_path)
            logging.info(f"Converted: {filename} → {os.path.basename(output_path)}")

        except Exception as e:
            logging.error(f"Failed to convert {filename}: {str(e)}")

    logging.info("Conversion complete.")

if __name__ == "__main__":
    main()
