import os
import numpy as np
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import rasterio
from pathlib import Path

# Settings
root_dir = "/path/to/root_directory_with_subfolders"
output_dir = "./lr_hr_pairs"
os.makedirs(output_dir, exist_ok=True)

HR_PATCH_SIZE = 1024
STRIDE = 1024
wavelet = 'haar'

# Initialize wavelet transforms
dwt = DWTForward(J=1, wave=wavelet, mode='zero')
idwt = DWTInverse(wave=wavelet, mode='zero')

def extract_patches(image, patch_size, stride):
    h, w = image.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches

# Recursive search for .nitf files with "_SICD_" in the name
sicd_nitf_files = [
    os.path.join(dp, f)
    for dp, _, filenames in os.walk(root_dir)
    for f in filenames
    if f.endswith(".nitf") and "_SICD_" in f
]

print(f"Found {len(sicd_nitf_files)} SICD NITF files.")

for fpath in sicd_nitf_files:
    try:
        with rasterio.open(fpath) as src:
            image = src.read(1)  # Read first band (SAR usually single-band)
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)  # normalize to [0, 1]
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
        continue

    hr_patches = extract_patches(image, HR_PATCH_SIZE, STRIDE)
    fname = Path(fpath).stem
    print(f"{fname}: extracted {len(hr_patches)} HR patches")

    for i, patch in enumerate(hr_patches):
        tensor_img = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        Yl, Yh = dwt(tensor_img)

        # Reconstruct and calculate MSE (optional)
        recon = idwt((Yl, Yh))
        mse = torch.mean((tensor_img - recon) ** 2).item()

        # Save as .npz
        out_name = f"{fname}_patch{i:03d}.npz"
        out_path = os.path.join(output_dir, out_name)
        np.savez_compressed(out_path, lr=Yl.squeeze().numpy(), hr=patch, mse=mse)
        print(f"Saved: {out_name} | MSE: {mse:.6f}")
