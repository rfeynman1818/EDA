import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sarpy.io.complex.converter import open_complex
from sarpy.io.product.converter import open_product
from sarpy.visualization.remap import Density

def _open_reader(p):
    r=t=None
    try: r=open_complex(p); t='SICD'
    except: pass
    if r is None:
        try: r=open_product(p); t='SIDD'
        except Exception as e: raise RuntimeError(f"Could not open {p}: {e}")
    return r,t

def _resize_to_target_spacing(img, reader, target_spacing):
    try:
        rr=float(reader.sicd_meta.Grid.Row.SS); rc=float(reader.sicd_meta.Grid.Col.SS)
    except Exception:
        return img  # spacing not available (e.g., SIDD) → skip
    sy=rr/float(target_spacing); sx=rc/float(target_spacing)
    nh, nw=int(round(img.shape[0]*sy)), int(round(img.shape[1]*sx))
    if nh<=0 or nw<=0 or (nh==img.shape[0] and nw==img.shape[1]): return img
    try:
        import cv2
        return cv2.resize(img, (nw,nh), interpolation=cv2.INTER_CUBIC)
    except Exception:
        # fallback if OpenCV isn't installed
        from PIL import Image
        return np.array(Image.fromarray(img).resize((nw,nh), resample=Image.BICUBIC))

def show_like_your_code(nitf_path, index=None, target_spacing=None, decim=1, figsize=(15,15)):
    reader,rtype=_open_reader(nitf_path)

    s=(slice(None,None,decim), slice(None,None,decim))
    if rtype=='SICD':
        A = reader[s] if index is None else reader[s+(index,)]
        img = Density()(np.abs(A))  # same remap you’re using
        if target_spacing is not None:
            img = _resize_to_target_spacing(img, reader, target_spacing)
        is_gray=True
    else:
        A = reader[s] if index is None else reader[s+(slice(None),index)]
        img = A if A.ndim==2 else (A[...,:3] if A.shape[-1]>=3 else A[...,0])
        is_gray = (img.ndim==2)

    fig,ax=plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray' if is_gray else None)
    ax.set_title(f"Image: {Path(nitf_path).name}"
                 + (f"  (TARGET_SPACING={target_spacing} m)" if (rtype=='SICD' and target_spacing) else "")
                 + (f"  decim={decim}" if decim and decim>1 else ""))
    ax.axis('off'); plt.tight_layout(); plt.show()
    return img
