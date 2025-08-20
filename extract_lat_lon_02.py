import argparse, os, re, xml.etree.ElementTree as ET, csv

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
    return None, None

def process_geojsons(geojsons, root, tokens, out_csv):
    prefixes = sorted(tokens, key=len, reverse=True)
    token_re = re.compile(rf'(({"|".join(map(re.escape,prefixes))})_\d+_)', re.IGNORECASE)

    rows=[]
    for gj in geojsons:
        gj = gj.strip()
        if not gj or not gj.lower().endswith('.geojson'): continue
        base=os.path.basename(gj)
        m = token_re.search(base)
        token = m.group(1) if m else ''
        target_dir = ''
        if token:
            dirs = [d for d in os.listdir(root) if d.startswith(token) and os.path.isdir(os.path.join(root,d))]
            target_dir = os.path.join(root, dirs[0]) if dirs else ''
        xml_name = os.path.splitext(base)[0] + '.xml'
        xml_path = os.path.join(target_dir, xml_name) if target_dir else ''
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

    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['geojson','prefix_token','matched_dir','xml_path','lat','lon'])
        w.writeheader(); w.writerows(rows)

    for r in rows:
        print(f"{os.path.basename(r['geojson'])}\t{r['prefix_token']}\t"
              f"{os.path.basename(r['matched_dir']) if r['matched_dir'] else ''}\t"
              f"{os.path.basename(r['xml_path']) if r['xml_path'] else ''}\t"
              f"{r['lat'] or ''}\t{r['lon'] or ''}")
    print(f"\nWrote CSV: {out_csv}")
    return rows

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('listfile', help='text file with list of .geojson paths, one per line')
    ap.add_argument('--root', default='.', help='directory containing SLEDP/SEDF/SLED dirs')
    ap.add_argument('--csv', default='llh_results.csv')
    ap.add_argument('--tokens', default='SLEDP,SEDF,SLED')
    a=ap.parse_args()

    with open(a.listfile) as f:
        geojsons = [line.strip() for line in f if line.strip()]
    token_list=[t.strip() for t in a.tokens.split(',') if t.strip()]
    process_geojsons(geojsons, a.root, token_list, a.csv)
