import numpy as np, matplotlib.pyplot as plt
from sarpy.io.complex.converter import open_complex
from sarpy.io.product.converter import open_product
from sarpy.visualization.remap import Density

def quicklook_nitf(path, index=None, target_max=4096, use_db=True, vclip=(2,98)):
    # open as SICD (complex) or SIDD/WBID (detected)
    reader, rtype = None, None
    try:
        reader=open_complex(path); rtype='SICD'
    except: pass
    if reader is None:
        try: reader=open_product(path); rtype='SIDD'
        except Exception as e: raise RuntimeError(f"Could not open {path}: {e}")
    
    # auto decimation to keep image manageable
    try: h,w=reader.data_size[:2]
    except: h,w=getattr(reader,'shape',(0,0))[:2]
    decim=max(1,int(np.ceil(max(h,w)/target_max))) if max(h,w)>0 else 1
    
    # slice out data
    s=(slice(None,None,decim), slice(None,None,decim))
    if rtype=='SICD':
        arr=reader[s] if index is None else reader[s+(index,)]
        amp=np.abs(arr)
        img=20*np.log10(amp+1e-12) if use_db else Density()(amp)
        lo,hi=np.percentile(img[np.isfinite(img)], vclip)
        plt.imshow(np.clip(img,lo,hi), cmap='gray')
        plt.title(f"SICD idx={index or 0} decim={decim}{' dB' if use_db else ''}")
    else:
        arr=reader[s] if index is None else reader[s+(slice(None),index)]
        if arr.ndim==2:  # grayscale
            lo,hi=np.percentile(arr[np.isfinite(arr)], vclip)
            plt.imshow(np.clip(arr,lo,hi), cmap='gray')
        elif arr.shape[-1]>=3:  # RGB
            lo,hi=np.percentile(arr, vclip)
            plt.imshow(np.clip(arr,lo,hi))
        else:
            lo,hi=np.percentile(arr[...,0], vclip)
            plt.imshow(np.clip(arr[...,0],lo,hi), cmap='gray')
        plt.title(f"SIDD idx={index or 0} decim={decim}")
    
    plt.axis('off'); plt.tight_layout(); plt.show()
