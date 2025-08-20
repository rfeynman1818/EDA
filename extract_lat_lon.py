import argparse, os, re, glob, sys, xml.etree.ElementTree as ET, csv

def strip_ns(tag): return tag.rsplit('}',1)[-1] if '}' in tag else tag

def find_llh_lat_lon(xml_path):
    try: root = ET.parse(xml_path).getroot()
    except Exception: return None, None
    for node in root.iter():
        if strip_ns(node.tag).lower() == 'llh':
            lat = lon = None
            for c in node.iter():
                t = strip_ns(c.tag).lower()
                if t == 'lat' and not lat: lat = (c.text or '').strip()
                if t == 'lon' and not lon: lon = (c.text or '').strip()
            return lat, lon
    lat = lon = None
    for c in root.iter():
        t = strip_ns(c.tag).lower()
        if t == 'lat' and not lat: lat = (c.text or '').strip()
        if t == 'lon' and not lon: lon = (c.text or '').strip()
    return lat, lon

def collect_geojsons(args):
    out=[]
    for p in args:
        if any(ch in p for ch in ['*','?','[']): out.extend(sorted(glob.glob(p)))
        elif os.path.isdir(p): out.extend(sorted(glob.glob(os.path.join(p,'*.geojson'))))
        else:
            if p.lower().endswith('.geojson') and os.path.exists(p): out.append(p)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--root', default='.', help='root to search prefixed directories (default: .)')
    ap.add_argument('--csv', default='llh_results.csv')
    ap.add_argument('--tokens', default='SLEDP,SEDF,SLED',
                    help='comma-separated allowed prefixes (default: SLEDP,SEDF,SLED)')
    a=ap.parse_args()

    prefixes = [t.strip() for t in a.tokens.split(',') if t.strip()]
    # Order by length desc so SLEDP is matched before SLED
    prefixes.sort(key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in prefixes)
    token_re = re.compile(rf'(({alt})_\d+_)', re.IGNORECASE)

    geojsons = collect_geojsons(a.inputs)
    if not geojsons:
        print('No .geojson files found from inputs', file=sys.stderr); sys.exit(1)

    rows=[]
    for gj in geojsons:
        base=os.path.basename(gj)
        m = token_re.search(base)
        token = m.group(1) if m else ''
        dir_candidates = sorted(
            d for d in glob.glob(os.path.join(a.root, token+'*'))
            if token and os.path.isdir(d)
        )
        target_dir = dir_candidates[0] if dir_candidates else ''
        xml_name = os.path.splitext(base)[0] + '.xml'
        xml_path = os.path.join(target_dir, xml_name) if target_dir else ''
        if not (xml_path and os.path.exists(xml_path)) and target_dir:
            alt_xml = glob.glob(os.path.join(target_dir, os.path.splitext(base)[0]+'*.xml'))
            xml_path = alt_xml[0] if alt_xml else ''
        lat, lon = (None, None)
        if xml_path and os.path.exists(xml_path):
            lat, lon = find_llh_lat_lon(xml_path)
        rows.append({
            'geojson': gj,
            'prefix_token': token,
            'matched_dir': target_dir,
            'xml_path': xml_path,
            'lat': lat, 'lon': lon
        })

    with open(a.csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['geojson','prefix_token','matched_dir','xml_path','lat','lon'])
        w.writeheader(); w.writerows(rows)

    for r in rows:
        print(f"{os.path.basename(r['geojson'])}\t{r['prefix_token']}\t"
              f"{os.path.basename(r['matched_dir']) if r['matched_dir'] else ''}\t"
              f"{os.path.basename(r['xml_path']) if r['xml_path'] else ''}\t"
              f"{r['lat'] or ''}\t{r['lon'] or ''}")
    print(f"\nWrote CSV: {a.csv}")

if __name__ == '__main__': main()
