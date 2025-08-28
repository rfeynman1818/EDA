# %matplotlib widget  # ← uncomment in a notebook cell for best interactivity

import numpy as np, matplotlib.pyplot as plt
from sarpy.io.complex.converter import open_complex
from sarpy.io.product.converter import open_product
from sarpy.visualization.remap import Density

def _open_reader(path):
    r=t=None
    try: r=open_complex(path); t='SICD'
    except: pass
    if r is None:
        try: r=open_product(path); t='SIDD'
        except Exception as e: raise RuntimeError(f"Could not open {path}: {e}")
    return r,t

def _auto_decim(reader, target_max):
    try: h,w=reader.data_size[:2]
    except: h,w=getattr(reader,'shape',(0,0))[:2]
    return max(1, int(np.ceil(max(h,w)/target_max))) if max(h,w)>0 else 1

def _read_display_array(reader, rtype, index, decim, use_db, vclip):
    s=(slice(None,None,decim), slice(None,None,decim))
    if rtype=='SICD':
        A = reader[s] if index is None else reader[s+(index,)]
        amp = np.abs(A)
        img = 20*np.log10(amp+1e-12) if use_db else Density()(amp)
        g = img[np.isfinite(img)]
        lo,hi = np.percentile(g, vclip) if g.size else (None,None)
        return np.clip(img, lo, hi) if (lo is not None) else img, True
    else:
        A = reader[s] if index is None else reader[s+(slice(None),index)]
        if A.ndim==2:
            g=A[np.isfinite(A)]
            lo,hi = np.percentile(g, vclip) if g.size else (None,None)
            return (np.clip(A,lo,hi) if lo is not None else A), True
        elif A.shape[-1]>=3:
            lo,hi=np.percentile(A, vclip)
            return np.clip(A, lo, hi), False
        else:
            g=A[...,0]
            lo,hi=np.percentile(g, vclip)
            return np.clip(g, lo, hi), True

class _ZoomPan:
    def __init__(self, ax):
        self.ax=ax; self.fig=ax.figure
        self._press=None
        self.cid_press=self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release=self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion=self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_scroll=self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes!=self.ax: return
        if event.button==1:  # left-drag to pan
            self._press=(event.xdata, event.ydata, *self.ax.get_xlim(), *self.ax.get_ylim())

    def on_release(self, event): self._press=None

    def on_motion(self, event):
        if self._press is None or event.inaxes!=self.ax: return
        x0,y0,x1,x2,y1,y2=self._press
        dx,dy=event.xdata-x0, event.ydata-y0
        self.ax.set_xlim(x1-dx, x2-dx)
        self.ax.set_ylim(y1-dy, y2-dy)
        self.ax.figure.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes!=self.ax: return
        # zoom centered on cursor
        xlim=self.ax.get_xlim(); ylim=self.ax.get_ylim()
        x,y=event.xdata, event.ydata
        scale=0.9 if event.button=='up' else 1/0.9
        def _zoom(lo, hi, c):
            span=(hi-lo)*scale
            return c-(c-lo)*scale, c+(hi-c)*scale
        self.ax.set_xlim(*_zoom(xlim[0], xlim[1], x))
        self.ax.set_ylim(*_zoom(ylim[0], ylim[1], y))
        self.ax.figure.canvas.draw_idle()

def quicklook_nitf_interactive(path, index=None, target_max=4096, use_db=True, vclip=(2,98)):
    reader,rtype=_open_reader(path)
    decim=_auto_decim(reader, target_max)
    img, is_gray=_read_display_array(reader, rtype, index, decim, use_db, vclip)

    fig,ax=plt.subplots(figsize=(8,6))
    if is_gray: ax.imshow(img, cmap='gray', interpolation='nearest')
    else: ax.imshow(img, interpolation='nearest')
    ax.set_title(f"{rtype} idx={index or 0}  decim={decim}" + ("  dB" if (rtype=='SICD' and use_db) else ""))
    ax.axis('off')
    _ZoomPan(ax)
    plt.show()

