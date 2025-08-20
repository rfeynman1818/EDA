import argparse, os, re, xml.etree.ElementTree as ET, csv, sys

def strip_ns(t): return t.rsplit('}',1)[-1] if '}' in t else t

def find_llh_lat_lon(p):
    try: r=ET.parse(p).getroot()
    except Exception: return None,None
    for n in r.iter():
        if strip_ns(n.tag).lower()=='llh':
            lat=lon=None
            for c in n.iter():
                t=strip_ns(c.tag).lower()
                if t=='lat' and not lat: lat=(c.text or '').strip()
                if t=='lon' and not lon: lon=(c.text or '').strip()
            return lat,lon
    return None,None

def read_listfile(lp):
    xs=[]
    try:
        with open(lp,'r') as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith('#'): continue
                s=os.path.expanduser(os.path.expandvars(s))
                xs.append(s)
    except FileNotFoundError:
        print(f'List file not found: {lp}', file=sys.stderr); sys.exit(1)
    return xs

def process(geojsons, root, prefixes, out_csv):
    prefixes=sorted([p for p in prefixes if p], key=len, reverse=True)
    token_re=re.compile(rf'(({"|".join(map(re.escape,prefixes))})_\d+_)', re.IGNORECASE)
    rows=[]
    for gj in geojsons:
        base=os.path.basename(gj)
        if not base.lower().endswith('.geojson') or not os.path.exists(gj):
            rows.append({'geojson':gj,'prefix_token':'','matched_dir':'','xml_path':'','lat':None,'lon':None}); continue
        m=token_re.search(base); token=m.group(1) if m else ''
        target_dir=''
        if token and os.path.isdir(root):
            ds=[d for d in os.listdir(root) if d.startswith(token) and os.path.isdir(os.path.join(root,d))]
            ds.sort(); target_dir=os.path.join(root,ds[0]) if ds else ''
        xml_name=os.path.splitext(base)[0]+'.xml'
        xml_path=os.path.join(target_dir,xml_name) if target_dir else ''
        lat=lon=None
        if xml_path and os.path.exists(xml_path): lat,lon=find_llh_lat_lon(xml_path)
        rows.append({'geojson':gj,'prefix_token':token,'matched_dir':target_dir,'xml_path':xml_path,'lat':lat,'lon':lon})
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['geojson','prefix_token','matched_dir','xml_path','lat','lon']); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{os.path.basename(r['geojson'])}\t{r['prefix_token']}\t{os.path.basename(r['matched_dir']) if r['matched_dir'] else ''}\t{os.path.basename(r['xml_path']) if r['xml_path'] else ''}\t{r['lat'] or ''}\t{r['lon'] or ''}")
    print(f"\nWrote CSV: {out_csv}")
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('listfile', help='text file with one .geojson path per line')
    ap.add_argument('--root', default='.', help='directory containing SLEDP_/SEDF_/SLED_ dirs')
    ap.add_argument('--csv', default='llh_results.csv', help='output CSV path')
    ap.add_argument('--tokens', default='SLEDP,SEDF,SLED', help='comma-separated prefixes')
    a=ap.parse_args()
    token_list=[t.strip() for t in a.tokens.split(',')]
    gj_list=read_listfile(a.listfile)
    process(gj_list, a.root, token_list, a.csv)

