import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def linear_remap(img, min_percent=1, max_percent=99):
    valid = img[~np.isnan(img)]
    if valid.size == 0:
        return np.zeros_like(img)
    low, high = np.percentile(valid, (min_percent, max_percent))
    return np.clip((img - low) / (high - low), 0, 1)

def load_tif_images_from_dir(directory):
    tifs = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.lower().endswith('.tif')]
    images = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else 0
            img[img == nodata] = np.nan
            remapped = linear_remap(img)
            images.append((remapped, os.path.basename(tif)))
    return images

def save_images_to_output(images, output_dir='output_images', dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, name) in enumerate(images):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        ax.imshow(img, cmap='gray', origin='upper')
        ax.set_title(name)
        ax.axis('off')
        filename = os.path.splitext(name)[0] + '.png'
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# Example usage:
# images = load_tif_images_from_dir('/path/to/tif_directory')
# save_images_to_output(images, output_dir='zoomable_grd_plots')
