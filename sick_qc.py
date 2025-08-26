import argparse, json, csv, sys, numpy as np
from pathlib import Path
from sarpy.io.general.utils import is_file_like
from sarpy.io.general.base import open_complex
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

def read_slc(path):
    rdr = open_complex(str(path))
    if rdr is None or not hasattr(rdr, "sicd_meta"):
        raise RuntimeError("Not a SICD/SLC readable file")
    return rdr, rdr.sicd_meta

def meta_checks(rdr, m):
    res = []
    ok = True
    def add(name, cond, msg=""):
        nonlocal ok; ok = ok and bool(cond); res.append({"check":name,"ok":bool(cond),"msg":msg})
    add("dims_match", rdr.shape==(m.ImageData.NumRows, m.ImageData.NumCols))
    add("ss_present", (m.Grid.Row.SS is not None) and (m.Grid.Col.SS is not None))
    add("pixel_type_complex", getattr(m, "ImageData", None) and m.ImageData.PixelType.upper().startswith("RE32"))
    for node in ("Grid","ImageFormation","GeoData"):
        add(f"{node}_present", getattr(m, node, None) is not None)
    return ok, res

def cfar_points(mag, guard=3, bg=12, k=6.0, max_pts=3000):
    f = maximum_filter(mag, size=2*guard+1)
    mask_peak = (mag==f)
    from scipy.ndimage import uniform_filter
    win = 2*(guard+bg)+1
    mu = uniform_filter(mag, win)
    mu2 = uniform_filter(mag**2, win)
    sigma = np.sqrt(np.maximum(mu2-mu**2, 1e-12))
    th = mu + k*sigma
    det = mask_peak & (mag>th)
    ys, xs = np.where(det)
    if ys.size>max_pts:
        idx = np.argsort(mag[ys, xs])[-max_pts:]
        ys, xs = ys[idx], xs[idx]
    return ys, xs

def impulse_metrics(win):
    a = np.abs(win)
    pr = a.mean(axis=1); pa = a.mean(axis=0)
    def line_metrics(p):
        peak = p.max(); i0 = int(p.argmax())
        hm = peak/np.sqrt(2)
        above = np.where(p>=hm)[0]
        if above.size<2: return np.nan, np.nan, np.nan
        res = above[-1]-above[0]+1
        q = p.copy(); q[i0]=0
        pslr = 20*np.log10(np.max(q)/(peak+1e-12)+1e-12)
        islr = 10*np.log10(np.sum(q**2)/((peak**2)+1e-12)+1e-12)
        return res, pslr, islr
    rr, prdb, irdb = line_metrics(pr); ra, padb, iadb = line_metrics(pa)
    return {"res_r_samp":rr,"PSLR_r_dB":prdb,"ISLR_r_dB":irdb,"res_a_samp":ra,"PSLR_a_dB":padb,"ISLR_a_dB":iadb}

def select_good_points(mag, ys, xs, w=33, border=64, min_sep=20):
    H,W = mag.shape
    keep = []
    grid = np.zeros_like(mag, dtype=bool)
    for y,x in np.argsort(-mag[ys,xs]):
        if y<w+border or x<w+border or y+w>=H-border or x+w>=W-border: continue
        if grid[y-min_sep:y+min_sep+1, x-min_sep:x+min_sep+1].any(): continue
        keep.append((y,x)); grid[y, x]=True
        if len(keep)>=200: break
    return keep

def robust_median(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan

def enl_on_homogeneous(intensity, tile=128, step=96, cv_th=0.25, min_tiles=10):
    H,W = intensity.shape
    enls=[]
    for y in range(0, H-tile+1, step):
        for x in range(0, W-tile+1, step):
            t = intensity[y:y+tile, x:x+tile]
            mu = t.mean(); sd = t.std()
            if mu<=0: continue
            cv = sd/(mu+1e-12)
            if cv<cv_th: enls.append((mu**2)/(sd**2+1e-12))
    enls = np.array(enls)
    if enls.size<min_tiles: return np.nan, 0
    return float(np.median(enls)), int(enls.size)

def nesz_estimate(intensity, frac=0.01):
    v = intensity.reshape(-1)
    v = v[np.isfinite(v) & (v>0)]
    if v.size<1000: return np.nan
    t = np.partition(v, int(frac*len(v)))[:int(frac*len(v))]
    return float(np.median(t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--window", type=int, default=33)
    args = ap.parse_args()
    rdr, m = read_slc(Path(args.input))
    rows, cols = rdr.shape
    row_ss, col_ss = m.Grid.Row.SS, m.Grid.Col.SS
    ok, meta_res = meta_checks(rdr, m)
    slc = rdr[::args.decimate, ::args.decimate]
    mag = np.abs(slc)
    ys, xs = cfar_points(mag, k=6.0)
    pts = select_good_points(mag, ys, xs, w=args.window)
    pt_metrics=[]
    for (y,x) in pts:
        win = slc[y-args.window:y+args.window+1, x-args.window:x+args.window+1]
        pt_metrics.append(impulse_metrics(win))
    res_r_m = [d["res_r_samp"] for d in pt_metrics]
    res_a_m = [d["res_a_samp"] for d in pt_metrics]
    pslr = [d["PSLR_r_dB"] for d in pt_metrics]+[d["PSLR_a_dB"] for d in pt_metrics]
    islr = [d["ISLR_r_dB"] for d in pt_metrics]+[d["ISLR_a_dB"] for d in pt_metrics]
    med = {
        "meta_ok": ok,
        "n_points_used": len(pt_metrics),
        "res_r_m": robust_median(res_r_m)*row_ss*args.decimate,
        "res_a_m": robust_median(res_a_m)*col_ss*args.decimate,
        "PSLR_dB_med": robust_median(pslr),
        "ISLR_dB_med": robust_median(islr),
    }
    inten = (np.abs(slc)**2).astype(np.float64)
    enl, ntiles = enl_on_homogeneous(inten)
    med["ENL_med"] = enl; med["ENL_tiles"] = ntiles
    med["NESZ_proxy"] = nesz_estimate(inten)
    predicted_res_r = getattr(getattr(m, "Grid", None), "Row", None).SS if hasattr(m.Grid,"Row") else None
    predicted_res_a = getattr(getattr(m, "Grid", None), "Col", None).SS if hasattr(m.Grid,"Col") else None
    med["row_ss_m"] = row_ss; med["col_ss_m"] = col_ss
    med["rows"] = rows; med["cols"] = cols
    print(json.dumps({"summary":med,"meta_checks":meta_res}, indent=2))
    if args.out_prefix:
        jp = Path(f"{args.out_prefix}.json"); cp = Path(f"{args.out_prefix}.csv")
        with open(jp,"w") as f: json.dump({"summary":med,"meta_checks":meta_res,"point_metrics":pt_metrics}, f, indent=2)
        with open(cp,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=pt_metrics[0].keys() if pt_metrics else ["res_r_samp","res_a_samp","PSLR_r_dB","PSLR_a_dB","ISLR_r_dB","ISLR_a_dB"])
            w.writeheader(); [w.writerow(d) for d in pt_metrics]

if __name__=="__main__": main()
