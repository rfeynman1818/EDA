#!/usr/bin/env python3
import argparse, os, csv, numpy as np
from matplotlib import pyplot as plt

def try_open(path):
    reader=None; rtype=None
    try:
        from sarpy.io.product.converter import open_product
        reader=open_product(path); rtype=getattr(reader,'reader_type','SIDD')
    except Exception: pass
    if reader is None:
        try:
            from sarpy.io.complex.converter import open_complex
            reader=open_complex(path); rtype=getattr(reader,'reader_type','SICD')
        except Exception: pass
    if reader is None:
        try:
            import rasterio
            src=rasterio.open(path)
            class RIO:
                reader_type='RIO'
                shape=(src.height,src.width,src.count)
                def __init__(self,src): self._src=src
                def __getitem__(self,slc):
                    if isinstance(slc,tuple) and len(slc)==3 and isinstance(slc[2],int):
                        r=slc[0]; c=slc[1]; b=slc[2]
                        data=self._src.read(b+1)[r,c]
                    else:
                        data=np.stack([self._src.read(i+1) for i in range(self._src.count)],-1)
                        if isinstance(slc,tuple): data=data[slc]
                    return data
            reader=RIO(src); rtype='GRD'
        except Exception as e:
            raise SystemExit(f"Could not open {path}: {e}")
    return reader, rtype

def read_arr(reader, rtype, index, band, decim, use_db):
    if rtype in ('SICD','SICDTypeReader','SICDReader'):
        from sarpy.visualization.remap import Density
        data = reader[::decim, ::decim, index] if index is not None else reader[::decim, ::decim]
        amp=np.abs(data)
        img=20*np.log10(amp+1e-12) if use_db else amp
        disp=Density()(img) if not use_db else img
        return np.asarray(disp), 'SICD'
    else:
        arr = reader[::decim, ::decim] if index is None else reader[::decim, ::decim, :, index]
        if arr.ndim==2: return np.asarray(arr), 'GRD'
        if band is not None: return np.asarray(arr[...,band]), 'GRD'
        return np.asarray(arr[...,0]), 'GRD'

def hist_metrics(img, bins=256):
    finite=np.isfinite(img); x=img[finite]
    if x.size==0: return None
    x=x.astype(np.float64)
    x=(x-x.min())/(x.max()-x.min()+1e-12)
    h,edges=np.histogram(x,bins=bins,range=(0,1),density=False)
    p=h.astype(np.float64)/max(1,h.sum())
    cdf=np.cumsum(p)
    lo=np.searchsorted(cdf,0.02); hi=np.searchsorted(cdf,0.98)
    p02, p98 = edges[lo], edges[min(hi,len(edges)-1)]
    spread=p98-p02
    mu=(np.arange(bins)+0.5)/bins
    mean=(p*mu).sum()
    var=(p*(mu-mean)**2).sum()
    std=np.sqrt(max(var,0))
    nz=(p>0).sum()/bins
    thr=otsu_threshold_from_hist(h)
    w0=p[:thr].sum(); w1=p[thr:].sum()
    m0=(p[:thr]*mu[:thr]).sum()/max(w0,1e-12); m1=(p[thr:]*mu[thr:]).sum()/max(w1,1e-12)
    interclass=w0*w1*(m1-m0)**2
    return dict(p02=p02,p98=p98,spread=spread,std=std,nonzero_frac=nz,otsu_bin=thr,interclass_var=interclass,mean=mean)

def otsu_threshold_from_hist(h):
    h=h.astype(np.float64); p=h/h.sum() if h.sum()>0 else h
    bins=len(h); omega=np.cumsum(p); mu=np.cumsum(p*np.arange(bins)); mu_t=mu[-1]
    sigma_b=(mu_t*omega-mu)**2/(omega*(1-omega)+1e-12)
    t=int(np.nanargmax(sigma_b))
    return t

def linear_stretch(img, p02, p98):
    x=img.astype(np.float64)
    lo,hi=np.percentile(x[np.isfinite(x)],[p02*100, p98*100]) if (0<=p02<p98<=1) else (np.min(x), np.max(x))
    y=(x-lo)/(hi-lo+1e-12); y=np.clip(y,0,1)
    return y

def assess_contrast(m):
    bad = (m['spread']<0.25 and m['std']<0.12) or (m['interclass_var']<1e-3 and m['nonzero_frac']<0.3)
    return 'Bad' if bad else 'Acceptable'

def plot_triptych(orig, stretched, hist, title, savepath=None, vmin=None, vmax=None):
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(orig, cmap='gray', vmin=vmin, vmax=vmax); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(stretched, cmap='gray', vmin=0, vmax=1); axs[1].set_title('Stretched'); axs[1].axis('off')
    axs[2].plot(hist); axs[2].set_title('Histogram'); axs[2].set_xlim(0,len(hist)); axs[2].set_yticks([])
    fig.suptitle(title); fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=150)
    plt.show()

def run_one(path, index, band, decim, vmin, vmax, use_db, outdir, save_figs, percentiles, quiet):
    reader,rtype=try_open(path)
    arr, kind = read_arr(reader, rtype, index, band, decim, use_db)
    m=hist_metrics(arr)
    if m is None: raise SystemExit("Empty/invalid image")
    y=linear_stretch(arr, m['p02'], m['p98']) if percentiles else (arr-arr.min())/(arr.max()-arr.min()+1e-12)
    m2=hist_metrics(y)
    verdict_before=assess_contrast(m)
    verdict_after=assess_contrast(m2)
    improved = (m2['spread']>m['spread']+0.05) or (m2['std']>m['std']+0.03)
    if not quiet:
        print(f"type={kind} shape={arr.shape} p02={m['p02']:.3f} p98={m['p98']:.3f} spread={m['spread']:.3f} std={m['std']:.3f} interclass={m['interclass_var']:.3e}")
        print(f"verdict_before={verdict_before} verdict_after={verdict_after} improved={improved}")
    if save_figs or not quiet:
        h,_=np.histogram(((arr-arr.min())/(arr.max()-arr.min()+1e-12)).ravel(),bins=256,range=(0,1))
        figpath=None
        if save_figs and outdir:
            os.makedirs(outdir, exist_ok=True)
            base=os.path.splitext(os.path.basename(path))[0]
            figpath=os.path.join(outdir, f"{base}_contrast_review.png")
        plot_triptych(arr, y, h, f"{os.path.basename(path)} [{kind}]", figpath, vmin, vmax)
    return dict(file=path, kind=kind, shape=str(arr.shape), p02=m['p02'], p98=m['p98'], spread=m['spread'], std=m['std'], interclass=m['interclass_var'], verdict_before=verdict_before, verdict_after=verdict_after, improved=bool(improved))

def main():
    ap=argparse.ArgumentParser(description="GRD/SICD visual contrast assessor")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--decim", type=int, default=1)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--save-figs", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--no-percentile-stretch", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a=ap.parse_args()
    rows=[]
    for p in a.paths:
        try:
            r=run_one(p,a.index,a.band,max(1,a.decim),a.vmin,a.vmax,a.db,a.outdir,a.save_figs,not a.no_percentile_stretch,a.quiet)
            rows.append(r)
        except Exception as e:
            rows.append(dict(file=p, error=str(e)))
            if not a.quiet: print(f"Error on {p}: {e}")
    if a.csv:
        cols=["file","kind","shape","p02","p98","spread","std","interclass","verdict_before","verdict_after","improved","error"]
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows:
                for k in cols:
                    if k not in r: r[k]=r.get(k,"")
                w.writerow(r)

if __name__=="__main__": main()
