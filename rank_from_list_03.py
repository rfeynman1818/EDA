#!/usr/bin/env python3
import argparse, csv, os, sys, glob, time, logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from sarpy.io.complex.converter import open_complex

# ---------------------------- Backend abstraction ---------------------------- #

class Backend:
    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                import cupy as xp
                from cupyx.scipy.ndimage import maximum_filter, uniform_filter
            except Exception as e:
                raise RuntimeError("GPU mode requested but CuPy/cupyx not available") from e
            self.xp = xp
            self.maximum_filter = maximum_filter
            self.uniform_filter = uniform_filter
        else:
            import numpy as xp
            from scipy.ndimage import maximum_filter, uniform_filter
            self.xp = xp
            self.maximum_filter = maximum_filter
            self.uniform_filter = uniform_filter

    def to_backend(self, a):
        return self.xp.asarray(a) if self.use_gpu else a

    def to_cpu(self, a):
        return self.xp.asnumpy(a) if self.use_gpu else a

# ------------------------------- Configuration ------------------------------- #

EPS = 1e-12
DEFAULT_EXTS = {".nitf", ".ntf", ".NITF", ".NTF"}

def setup_logger(verbosity: int, logfile: Optional[str]) -> logging.Logger:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logger = logging.getLogger("nitf_ranker")
    logger.setLevel(level)
    logger.handlers[:] = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# --------------------------------- Utilities --------------------------------- #

def load_filelist(txt_path: Path, logger: logging.Logger, exts: Optional[set]=DEFAULT_EXTS) -> List[Path]:
    base = Path(txt_path).resolve().parent
    files, missing = [], 0
    logger.info("[1/4] Loading file list from %s", txt_path)
    with open(txt_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"): continue
            s = os.path.expandvars(os.path.expanduser(s))
            p = Path(s)
            if not p.is_absolute(): p = base / p
            if any(c in str(p) for c in "*?[]"):
                hits = [Path(h) for h in glob.glob(str(p), recursive=True)]
                files.extend(hits)
            else:
                if p.exists(): files.append(p)
                else:
                    missing += 1
                    logger.warning("Missing path from list: %s", p)
    if exts:
        files = [Path(x).resolve() for x in files if Path(x).suffix in exts]
    files = sorted(dict.fromkeys(files))
    logger.info("[1/4] Found %d files (%d missing lines ignored).", len(files), missing)
    return files

def read_slc(path: Path, decimate: int) -> "np.ndarray":
    rdr = open_complex(str(path))
    if rdr is None:
        raise RuntimeError("open_complex returned None")
    try:
        return rdr[::decimate, ::decimate]
    finally:
        del rdr

# --------------------------------- Metrics ----------------------------------- #

def cfar_count(a_mag, backend: Backend, guard=2, bg=8, k=8.0) -> int:
    xp = backend.xp
    win = 2 * (guard + bg) + 1
    mu = backend.uniform_filter(a_mag, win)
    mu2 = backend.uniform_filter(a_mag * a_mag, win)
    sig = xp.sqrt(xp.maximum(mu2 - mu * mu, EPS))
    th = mu + k * sig
    locmax = (a_mag == backend.maximum_filter(a_mag, size=2 * guard + 1))
    return int(xp.count_nonzero(locmax & (a_mag > th)))

@dataclass
class Metrics:
    dr_db: float
    peak_db: float
    cfar: int
    noise_proxy: float
    score: float
    bad_flag: bool
    error: str = ""

def compute_metrics(slc_cpu, backend: Backend, dr_th=20.0, cfar_th=50, peak_th=25.0) -> Metrics:
    xp = backend.xp
    a = xp.abs(backend.to_backend(slc_cpu))
    i = (a * a).astype(xp.float64)

    p1 = xp.percentile(i, 1)
    p999 = xp.percentile(i, 99.9)
    dr_db = float(10.0 * xp.log10((p999 + EPS) / (p1 + EPS)))

    med_a = xp.median(a)
    peak_db = float(20.0 * xp.log10((a.max() + EPS) / (med_a + EPS)))

    # median of top ~1% intensities
    n = i.size
    k = max(1, int(0.01 * n))
    part = xp.partition(i.reshape(-1), n - k)[n - k :]
    nz = float(xp.median(part))

    cfar = cfar_count(a, backend, k=8.0)

    score = float(0.55 * dr_db + 0.3 * peak_db + 0.15 * max(0.0, float(xp.log10(cfar + 1))) * 20.0)
    bad_flag = (dr_db < dr_th) and (cfar < cfar_th) and (peak_db < peak_th)
    return Metrics(dr_db, peak_db, cfar, nz, score, bad_flag)

# --------------------------------- Pipeline ---------------------------------- #

def process_one(path: Path, decimate: int, backend: Backend, thresholds: Tuple[float,int,float], logger: Optional[logging.Logger]=None) -> dict:
    t0 = time.perf_counter()
    try:
        slc = read_slc(path, decimate)
        m = compute_metrics(slc, backend, *thresholds)
        row = dict(file=str(path), **asdict(m))
        status = "ok"
    except Exception as e:
        row = dict(file=str(path), dr_db=float("nan"), peak_db=float("nan"),
                   cfar=0, noise_proxy=float("nan"), score=-1e9, bad_flag=True, error=str(e))
        status = "err"
    dt = time.perf_counter() - t0
    if logger:
        if status == "ok":
            logger.debug("Processed %s in %.2fs | score=%.2f", path.name, dt, row["score"])
        else:
            logger.error("Error on %s in %.2fs: %s", path.name, dt, row["error"])
    row["_elapsed_s"] = dt
    return row

def rank_and_write(rows: List[dict], csv_path: Path, logger: logging.Logger):
    rows.sort(key=lambda r: r.get("score", -1e9), reverse=True)
    logger.info("[4/4] Writing CSV → %s", csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","file","score","dr_db","peak_db","cfar","noise_proxy","bad_flag","error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            out = {**r, "rank": k}
            out.pop("_elapsed_s", None)
            w.writerow(out)

# ----------------------------------- CLI ------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Batch rank NITF SLC/SICD quality with optional GPU acceleration.")
    ap.add_argument("list_txt", help="Text file: one path per line (globs allowed).")
    ap.add_argument("--csv", default="ranked_quality.csv")
    ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("--benchmark", type=int, default=0, help="process only first N files")
    ap.add_argument("--dr_th", type=float, default=20.0)
    ap.add_argument("--cfar_th", type=int, default=50)
    ap.add_argument("--peak_th", type=float, default=25.0)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0, help="CPU workers (0=auto; 1=sequential). Ignored in GPU mode.")
    ap.add_argument("--gpu", action="store_true", help="Use CuPy/cupyx on GPU.")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="-v, -vv for more logs; -q to quiet")
    ap.add_argument("-q", "--quiet", action="count", default=0)
    ap.add_argument("--logfile", default=None)
    args = ap.parse_args()

    verbosity = max(0, args.verbose - args.quiet)
    logger = setup_logger(verbosity, args.logfile)

    backend = Backend(use_gpu=args.gpu)
    logger.info("Backend: %s", "GPU (CuPy)" if args.gpu else "CPU (NumPy)")
    thresholds = (args.dr_th, args.cfar_th, args.peak_th)

    files = load_filelist(Path(args.list_txt), logger)
    if not files:
        logger.error("No files to process. Exiting.")
        sys.exit(1)

    if args.benchmark and args.benchmark < len(files):
        files = files[: args.benchmark]
        logger.info("[bench] Restricting to first %d files for timing.", len(files))

    logger.info("[2/4] Starting processing of %d files; decimate=%d", len(files), args.decimate)

    rows = []
    t_all = time.perf_counter()
    if args.gpu:
        # GPU: process sequentially to avoid device thrash; keep host->device copies small due to decimation
        for idx, f in enumerate(files, 1):
            if (idx % args.progress_every) == 0 or args.progress_every == 1:
                logger.info("[%3d/%d] %s", idx, len(files), f.name)
            rows.append(process_one(f, args.decimate, backend, thresholds, logger))
    else:
        # CPU: parallelize work; IO + compute
        from concurrent.futures import ProcessPoolExecutor, as_completed
        workers = (os.cpu_count() or 2) if args.workers == 0 else max(1, args.workers)
        if workers == 1:
            for idx, f in enumerate(files, 1):
                if (idx % args.progress_every) == 0 or args.progress_every == 1:
                    logger.info("[%3d/%d] %s", idx, len(files), f.name)
                rows.append(process_one(f, args.decimate, backend, thresholds, logger))
        else:
            logger.info("Using %d CPU workers", workers)
            # Rebuild lightweight backend in workers (CPU only)
            def _worker(path, decimate, thresholds):
                b = Backend(use_gpu=False)
                return process_one(Path(path), decimate, b, thresholds, None)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, str(f), args.decimate, thresholds): f for f in files}
                for k, fut in enumerate(as_completed(futs), 1):
                    f = futs[fut]
                    if (k % args.progress_every) == 0 or args.progress_every == 1:
                        logger.info("[%3d/%d] %s", k, len(files), f.name)
                    rows.append(fut.result())

    elapsed = time.perf_counter() - t_all
    avg = sum(r["_elapsed_s"] for r in rows) / len(rows)
    logger.info("[3/4] Done processing. Wall-time %.2fs | mean per-image %.2fs", elapsed, avg)

    rank_and_write(rows, Path(args.csv), logger)

    logger.info("Top 5 by score:")
    for r in rows[:5]:
        logger.info("  %6.2f | %s", r.get("score", -1e9), r["file"])
    bads = [r for r in rows if r.get("bad_flag")]
    if bads:
        logger.info("Examples flagged as grainy/dark:")
        for r in bads[:5]:
            logger.info("  %s (DR=%.1f dB, CFAR=%d, Peak=%.1f dB)", r["file"], r["dr_db"], r["cfar"], r["peak_db"])

if __name__ == "__main__":
    main()
