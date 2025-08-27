# What the script reports

## Runtime output (printed to the console)

* **\[1/4] Loading file list…** reads your `.txt` of paths (globs allowed) and warns about missing files.
* **\[2/4] Starting processing…** begins scoring; for each file you’ll see:

  * `[#/N] name …`
  * A timing line like `⏱ 0.82s (avg 0.91s/image over K files)`.
* **\[3/4] Ranking results…** sorts by quality score.
* **\[4/4] Writing CSV → …** writes the results.
* Finishes with **Top 5 by score** and several examples **flagged as “grainy/dark.”**

> Tip: `--benchmark N` processes only the first *N* files so you can measure throughput on your machine.

## CSV contents (sorted best → worst)

Each row corresponds to one image:

* **rank** — 1 = best score.
* **file** — absolute path.
* **score** *(higher = better)* — a weighted combo of the metrics below:

  ```
  score = 0.55·dr_db + 0.30·peak_db + 0.15·(20·log10(cfar+1), clipped at ≥0)
  ```
* **dr\_db** — dynamic range (dB) from intensity percentiles `p99.9/p1`.
  Higher ⇒ stronger contrast; very low ⇒ flat/noisy scene.
* **peak\_db** — peakiness (dB) `max(amplitude)/median(amplitude)`.
  Higher ⇒ distinct bright points; low ⇒ nothing stands out.
* **cfar** — count of strong local maxima from a CFAR-like detector (k=8).
  Higher ⇒ more distinct scatterers/structure.
* **noise\_proxy** — median of the lowest 1% of intensity pixels.
  Higher ⇒ higher background floor (noisier image).
* **bad\_flag** — `True` if it matches the “grainy/dark” look (defaults: `dr_db < 20`, `cfar < 50`, `peak_db < 25`).
  Tune with `--dr_th`, `--cfar_th`, `--peak_th`.
* **error** — error message if opening/reading failed. Errored rows get `score = -1e9` so they sink to the bottom and `bad_flag = True`.

## What you learn at a glance

* Which images are **best** (top ranks) vs **suspect** (low score or `bad_flag=True`).
* Whether the set has a **raised noise floor** (`noise_proxy` high), **weak contrast** (`dr_db` low), or **few strong targets** (`cfar` low).
* Your **actual throughput** from the ⏱ lines, so you can plan full runs or adjust settings.

## Notes & useful knobs

* **Speed/accuracy trade-off:** `--decimate` downsamples on read; larger values run faster but may slightly reduce `cfar`/peakiness.
* **Flag strictness:** adjust `--dr_th`, `--cfar_th`, `--peak_th` to match your “grainy/dark” appearance.
* **Quick timing:** use `--benchmark N` to time a subset before running all files.
