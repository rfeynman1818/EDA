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

def show_image_sequence(images):
    fig, ax = plt.subplots(figsize=(10, 10))
    idx = [0]

    def update():
        ax.clear()
        img, title = images[idx[0]]
        ax.imshow(img, cmap='gray', origin='upper')
        ax.set_title(f'{title} ({idx[0]+1}/{len(images)})')
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(images)
            update()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(images)
            update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()

# Example usage:
# images = load_tif_images_from_dir('/path/to/tif_directory')
# show_image_sequence(images)
