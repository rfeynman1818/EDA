# preds_geojson_to_coco.py
import json, argparse, glob, os
from pathlib import Path

def ring_bbox(r):
    xs=[float(x) for x,_ in r]; ys=[float(y) for _,y in r]
    x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
    return [x1,y1,max(0.0,x2-x1),max(0.0,y2-y1)]

def image_key_from_meta(meta):
    src=(meta or {}).get("source",{})
    img=(src or {}).get("image",{})
    return img.get("id") or src.get("filename")

def first_ring(geom):
    ic=(geom or {}).get("image_coordinates")
    if ic:
        if ic and ic[0] and isinstance(ic[0][0], (int,float)): return ic
        if ic and ic[0] and isinstance(ic[0][0], list) and ic[0][0] and isinstance(ic[0][0][0], (int,float)): return ic[0]
        if isinstance(ic[0], list): return ic[0]
    coords=(geom or {}).get("coordinates") or []
    t=(geom or {}).get("type","")
    if t=="Polygon" and coords: return coords[0]
    if t=="MultiPolygon" and coords and coords[0]: return coords[0][0]
    return None

def load_features(path):
    obj=json.load(open(path))
    if isinstance(obj, dict) and obj.get("type")=="FeatureCollection":
        return obj.get("features",[]), obj.get("metadata",{})
    if isinstance(obj, dict) and obj.get("type")=="Feature":
        return [obj], obj.get("metadata",{})
    # fallback: unknown wrapper → treat as no features
    return [], {}

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="in_path", help="Single preds .geojson/.json")
    g.add_argument("--in_dir", help="Directory containing many per-image .geojson files")
    ap.add_argument("--glob", default="*.geojson", help="Glob used with --in_dir (default: *.geojson)")
    ap.add_argument("--out", dest="out_dir", default="out_pred")
    ap.add_argument("--class_key", choices=["classification","label","auto"], default="auto")
    ap.add_argument("--image_key", choices=["metadata_id","metadata_filename","filename","basename"], default="basename",
                    help="How to form the image key for each file/feature")
    args=ap.parse_args()

    # gather files
    files=[]
    if args.in_path:
        files=[args.in_path]
    else:
        files=sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not files:
        raise SystemExit("[preds→coco] No input files found.")

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cat2id={}; next_cid=1
    img2id={}; next_img=1
    preds=[]

    def get_cls(props):
        if args.class_key=="classification": return props.get("classification")
        if args.class_key=="label": return props.get("label")
        return props.get("classification") if props.get("classification") is not None else props.get("label")

    def image_key_for_file(meta, file_path):
        if args.image_key=="metadata_id":
            return image_key_from_meta(meta) or os.path.basename(file_path)
        if args.image_key=="metadata_filename":
            return (meta.get("source",{}) or {}).get("filename") or os.path.basename(file_path)
        if args.image_key=="filename":
            return file_path  # full path string
        if args.image_key=="basename":
            return os.path.basename(file_path)

    for fp in files:
        feats, file_meta = load_features(fp)
        # default image key per file (can be overridden per-feature if feature.metadata present)
        default_key=image_key_for_file(file_meta, fp)
        if default_key not in img2id:
            img2id[default_key]=next_img; next_img+=1
        for f in feats:
            props=f.get("properties",{}); geom=f.get("geometry",{}); meta=f.get("metadata") or file_meta
            # choose image key (prefer feature metadata, fallback to file-based)
            ik = image_key_for_file(meta, fp) or default_key
            if ik not in img2id:
                img2id[ik]=next_img; next_img+=1
            img_id=img2id[ik]

            cls=get_cls(props)
            if cls not in cat2id: cat2id[cls]=next_cid; next_cid+=1
            cid=cat2id[cls]

            r=first_ring(geom)
            if not r or len(r)<3: 
                # optional fallback: XYXY bbox in properties
                bb=props.get("bbox")
                if bb and props.get("bbox_order","").upper()=="XYXY" and len(bb)==4:
                    x1,y1,x2,y2=map(float,bb); r=[[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]
                else:
                    continue
            x,y,w,h=ring_bbox(r)
            if w<=0 or h<=0: continue

            score=float(props.get("confidence",1.0))
            preds.append({"image_id":img_id,"category_id":cid,"bbox":[x,y,w,h],"score":score})

    json.dump(preds, open(out_dir/"preds.json","w"), indent=2)
    json.dump({"image_id_map": img2id, "category_id_map": {str(k):v for k,v in cat2id.items()}},
              open(out_dir/"mappings.json","w"), indent=2)
    print(f"[preds→coco] files:{len(files)} detections:{len(preds)} "
          f"→ {out_dir}/preds.json; maps → {out_dir}/mappings.json")

if __name__=="__main__":
    main()

