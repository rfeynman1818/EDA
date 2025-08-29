#!/usr/bin/env python3
import os, re, glob, argparse

def run(txtfile, base, outfile):
    print(f"[INFO] Reading entries from: {txtfile}")
    print(f"[INFO] Base search directory: {base}")
    print(f"[INFO] Output file will be: {outfile}")

    found_paths = []

    with open(txtfile) as f:
        for idx, raw in enumerate(f, 1):
            entry = raw.strip().strip('"')
            if not entry:
                print(f"[WARN] Line {idx} is empty. Skipping.")
                continue

            print(f"\n[STEP] Processing line {idx}: {entry}")

            # Extract prefix immediately after SICD_
            m = re.search(r'SICD_([A-Za-z0-9]+)_(\d+)_', entry)
            if not m:
                print(f"[SKIP] Could not extract SICD_*_*_ pattern from: {entry}")
                continue

            prefix, ident = m.group(1), m.group(2)
            pattern = f"{prefix}_{ident}_*"
            print(f"[INFO] Derived directory pattern: {pattern}")

            dirs = glob.glob(os.path.join(base, pattern))
            if not dirs:
                print(f"[MISS] No directory found for {entry} under {base} matching {pattern}")
                continue

            dirpath = sorted(dirs)[0]
            print(f"[INFO] Using directory: {dirpath}")

            target = os.path.join(dirpath, entry + ".nitf")
            if os.path.isfile(target):
                print(f"[FOUND] Direct hit: {target}")
                found_paths.append(target)
                continue

            print(f"[SEARCH] .nitf not directly found. Walking directory {dirpath}...")
            found = None
            for root, _, files in os.walk(dirpath):
                fn = entry + ".nitf"
                if fn in files:
                    found = os.path.join(root, fn)
                    break

            if found:
                print(f"[FOUND] Located via walk: {found}")
                found_paths.append(found)
            else:
                print(f"[MISS] .nitf not found for {entry} in {dirpath}")

    # Write results to output file
    print(f"\n[INFO] Writing {len(found_paths)} found paths to {outfile}")
    with open(outfile, "w") as out:
        for p in found_paths:
            out.write(p + "\n")
    print("[DONE] Script finished successfully.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", help="path to input .txt list")
    ap.add_argument("--base", default="/mnt/deutchland/sofos-skywatch-vision-lm-gov-efs/sup_training_dataset/SRF-2198",
                    help="base directory containing the subfolders")
    ap.add_argument("--out", default="found_paths.txt", help="output .txt file to write results")
    a = ap.parse_args()
    run(a.txt, a.base, a.out)
