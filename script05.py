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
    imshow_obj = [None]
    press = [None]

    def update_image():
        ax.clear()
        img, title = images[idx[0]]
        imshow_obj[0] = ax.imshow(img, cmap='gray', origin='upper')
        ax.set_title(f'{title} ({idx[0]+1}/{len(images)})')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % len(images)
            update_image()
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % len(images)
            update_image()

    def on_scroll(event):
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xrange = (cur_xlim[1] - cur_xlim[0])
        yrange = (cur_ylim[1] - cur_ylim[0])
        scale_factor = 1.2 if event.button == 'up' else 1/1.2
        new_xlim = [xdata - (xdata - cur_xlim[0]) * scale_factor,
                    xdata + (cur_xlim[1] - xdata) * scale_factor]
        new_ylim = [ydata - (ydata - cur_ylim[0]) * scale_factor,
                    ydata + (cur_ylim[1] - ydata) * scale_factor]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button != 1: return
        press[0] = (event.x, event.y, ax.get_xlim(), ax.get_ylim())

    def on_release(event):
        press[0] = None

    def on_motion(event):
        if press[0] is None or event.button != 1: return
        xpress, ypress, xlim0, ylim0 = press[0]
        dx = event.x - xpress
        dy = event.y - ypress
        inv = ax.transData.inverted()
        dx_data, dy_data = inv.transform((0, 0)) - inv.transform((dx, dy))
        ax.set_xlim(xlim0[0] + dx_data, xlim0[1] + dx_data)
        ax.set_ylim(ylim0[0] + dy_data, ylim0[1] + dy_data)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    update_image()
    plt.show()

# Example usage:
# images = load_tif_images_from_dir('/path/to/tif_directory')
# show_image_sequence(images)
