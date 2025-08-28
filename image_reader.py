#!/usr/bin/env python3
import argparse, numpy as np
from matplotlib import pyplot as plt

def main(path, index, band, decim, vmin, vmax, use_db):
    reader = None
    rtype = None
    try:
        from sarpy.io.complex.converter import open_complex
        reader = open_complex(path)  # SICD/SLC-like complex (often NITF)  :contentReference[oaicite:1]{index=1}
        rtype = 'SICD'
    except Exception:
        pass
    if reader is None:
        try:
            from sarpy.io.product.converter import open_product
            reader = open_product(path)  # SIDD/WBID products (often NITF)  :contentReference[oaicite:2]{index=2}
            rtype = 'SIDD'
        except Exception as e:
            raise SystemExit(f"Could not open {path} with SarPy complex or product readers: {e}")

    print(f"reader_type={getattr(reader,'reader_type',rtype)} size={getattr(reader,'data_size',None)}")

    if getattr(reader,'reader_type',rtype) == 'SICD':
        from sarpy.visualization.remap import Density  # tone mapping for complex SAR  :contentReference[oaicite:3]{index=3}
        remap = Density()
        data = reader[::decim, ::decim, index] if index is not None else reader[::decim, ::decim]
        amp = np.abs(data)
        img = 20*np.log10(amp+1e-12) if use_db else amp
        disp = remap(img) if not use_db else img
        plt.imshow(disp, cmap='gray', vmin=vmin, vmax=vmax); plt.title(f"SICD index={index or 0}")
    else:  # SIDD/WBID-like (detected image)
        arr = reader[::decim, ::decim] if index is None else reader[::decim, ::decim, :, index]
        if arr.ndim == 2:  # single-band
            plt.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax); plt.title(f"SIDD gray index={index or 0}")
        else:
            if band is not None:  # display a selected band
                plt.imshow(arr[..., band], cmap='gray', vmin=vmin, vmax=vmax); plt.title(f"SIDD band {band} index={index or 0}")
            elif arr.shape[-1] >= 3:  # assume RGB order for 3+
                plt.imshow(arr[..., :3]); plt.title(f"SIDD RGB index={index or 0}")
            else:
                plt.imshow(arr[..., 0], cmap='gray', vmin=vmin, vmax=vmax); plt.title(f"SIDD first band index={index or 0}")

    plt.axis('off'); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick viewer for SAR NITF via SarPy")
    p.add_argument("path")
    p.add_argument("--index", type=int, default=None, help="image index for multi-image NITF")
    p.add_argument("--band", type=int, default=None, help="band for SIDD (e.g., 0=gray or select RGB band)")
    p.add_argument("--decim", type=int, default=1, help="decimation factor (>=1)")
    p.add_argument("--vmin", type=float, default=None); p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--db", action="store_true", help="show complex magnitude in dB (SICD only)")
    a = p.parse_args()
    main(a.path, a.index, a.band, max(1,a.decim), a.vmin, a.vmax, a.db)
