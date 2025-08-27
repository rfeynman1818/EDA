import argparse, csv, os, sys, glob, time, numpy as np
from pathlib import Path
from sarpy.io.complex.converter import open_complex
from scipy.ndimage import maximum_filter, uniform_filter

eps = 1e-12

def load_filelist(txt_path, exts={".nitf",".ntf",".NITF",".NTF"}):
    print(f"[1/4] Loading file list from: {txt_path}", flush=True)
    base = Path(txt_path).resolve().parent
    files, missing = [], 0
    with open(txt_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"): continue
            s = os.path.expandvars(os.path.expanduser(s))
            p = Path(s)
            if not p.is_absolute(): p = (base / p)
            if any(c in str(p) for c in "*?[]"):
                hits = [Path(h) for h in glob.glob(str(p), recursive=True)]
                files.extend(hits)
            else:
                if p.exists(): files.append(p)
                else:
                    missing += 1
                    print(f"   ! Missing: {p}", flush=True)
    files = [Path(x).resolve() for x in files if (exts is None or Path(x).suffix in exts)]
    files = sorted(dict.fromkeys(files))
    print(f"[1/4] Found {len(files)} files ({missing} missing lines ignored).", flush=True)
    return files

def read_slc(path, dec=4):
    rdr = open_complex(str(path))
    if rdr is None: raise RuntimeError("open_complex returned None")
    try:
        slc = rdr[::dec, ::dec]
    finally:
        del rdr
    return slc

def cfar_count(mag, guard=2, bg=8, k=8.0):
    win = 2*(guard+bg)+1
    mu  = uniform_filter(mag, win)
    mu2 = uniform_filter(mag**2, win)
    sig = np.sqrt(np.maximum(mu2 - mu**2, eps))
    th  = mu + k*sig
    locmax = (mag == maximum_filter(mag, size=2*guard+1))
    return int(np.count_nonzero(locmax & (mag > th)))

def metrics(slc):
    a = np.abs(slc)
    i = (a*a).astype(np.float64)
    p1, p999 = np.percentile(i, 1), np.percentile(i, 99.9)
    dr_db = 10*np.log10((p999+eps)/(p1+eps))
    peak_db = 20*np.log10((a.max()+eps)/(np.median(a)+eps))
    npx = cfar_count(a, k=8.0)
    nz = np.median(np.partition(i.reshape(-1), max(1,int(0.01*i.size))-1)[:max(1,int(0.01*i.size))])
    return dict(dr_db=float(dr_db), peak_db=float(peak_db), cfar=int(npx), noise_proxy=float(nz))

def score(m):
    return float(0.55*m["dr_db"] + 0.3*m["peak_db"] + 0.15*max(0, np.log10(m["cfar"]+1))*20)

def looks_bad(m, dr_th=20, cfar_th=50, peak_th=25):
    return (m["dr_db"] < dr_th) and (m["cfar"] < cfar_th) and (m["peak_db"] < peak_th)

def main():
    ap = argparse.ArgumentParser(description="Batch rank NITF SLC/SICD quality from a text file of paths.")
    ap.add_argument("list_txt", help="Text file: one path per line (globs allowed).")
    ap.add_argument("--csv", default="ranked_quality.csv")
    ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("--benchmark", type=int, default=0, help="process only the first N files to measure speed, then exit")
    ap.add_argument("--dr_th", type=float, default=20.0, help="bad-flag DR threshold (dB)")
    ap.add_argument("--cfar_th", type=int, default=50, help="bad-flag CFAR peak count threshold")
    ap.add_argument("--peak_th", type=float, default=25.0, help="bad-flag peakiness threshold (dB)")
    ap.add_argument("--progress_every", type=int, default=1, help="print per-file progress every N files")
    args = ap.parse_args()

    files = load_filelist(args.list_txt)
    if not files:
        print("No files to process. Exiting.", flush=True); sys.exit(1)
    if args.benchmark and args.benchmark < len(files):
        files = files[:args.benchmark]
        print(f"[bench] Restricting to first {len(files)} files for timing.", flush=True)

    rows = []; durations=[]
    print(f"[2/4] Starting processing ({len(files)} files), decimate={args.decimate}", flush=True)
    for idx, f in enumerate(files, 1):
        if (idx % args.progress_every) == 0 or args.progress_every == 1:
            print(f"  [{idx:>3}/{len(files)}] {f.name} …", flush=True)
        t0 = time.perf_counter()
        try:
            slc = read_slc(f, args.decimate)
            m = metrics(slc)
            m["score"] = score(m)
            m["bad_flag"] = looks_bad(m, args.dr_th, args.cfar_th, args.peak_th)
            rows.append(dict(file=str(f), **m))
        except Exception as e:
            print(f"     ! Error on {f}: {e}", flush=True)
            rows.append(dict(file=str(f), error=str(e), dr_db=np.nan, peak_db=np.nan, cfar=0, noise_proxy=np.nan, score=-1e9, bad_flag=True))
        finally:
            dt = time.perf_counter() - t0
            durations.append(dt)
            avg = float(np.mean(durations))
            print(f"      ⏱ {dt:.2f}s (avg {avg:.2f}s/image over {len(durations)} files)", flush=True)

    print(f"[3/4] Ranking results…", flush=True)
    rows.sort(key=lambda r: r.get("score", -1e9), reverse=True)

    print(f"[4/4] Writing CSV → {args.csv}", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","file","score","dr_db","peak_db","cfar","noise_proxy","bad_flag","error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            r = {**r}
            r["rank"] = k
            if "error" not in r: r["error"] = ""
            w.writerow(r)

    print("Done.\nTop 5 by score:")
    for r in rows[:5]:
        print(f"  {r.get('score',-1e9):6.2f}  |  {r['file']}")
    print("Examples flagged as 'grainy/dark':")
    bads = [r for r in rows if r.get("bad_flag")]
    for r in bads[:5]:
        print(f"  {r['file']}  (DR={r['dr_db']:.1f} dB, CFAR={r['cfar']}, Peak={r['peak_db']:.1f} dB)")

if __name__ == "__main__":
    main()
