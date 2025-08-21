import os, re, sys, argparse, datetime

def find_date_in_path(p):
    m=[int(x) for x in re.findall(r'20\d{6}', p)]
    return max(m) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ext", default=".geojson")
    ap.add_argument("--dry-run", action="store_true")
    args=ap.parse_args()

    groups={}
    for dirpath,_,files in os.walk(args.root):
        for f in files:
            if not f.endswith(args.ext): continue
            base=f
            full=os.path.join(dirpath,f)
            d=find_date_in_path(full)
            if d is None: d=int(datetime.datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y%m%d"))
            groups.setdefault(base,[]).append((d,full))

    print(f"Found {len(groups)} unique filenames. Checking for duplicates...")
    sep="-"*62
    kept,deleted=0,0
    for base,items in sorted(groups.items()):
        if len(items)<2: continue
        items.sort(key=lambda x:(x[0],x[1]))
        keep=items[-1][1]
        dels=[p for _,p in items[:-1]]
        print(sep)
        print(f"Found {len(items)} copies of: {base}")
        print(f"✅  Keeping: {keep}")
        for p in dels:
            print(f"🗑️  Deleting: {p}")
            if not args.dry_run:
                try: os.remove(p); deleted+=1
                except Exception as e: print(f"   ! Failed to delete {p}: {e}")
        kept+=1
    print(sep)
    print("✨ Script finished.")
    print(f"Groups processed: {kept} | Files deleted: {deleted} | Dry run: {args.dry_run}")

if __name__=="__main__": main()
