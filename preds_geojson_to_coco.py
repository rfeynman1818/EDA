# preds_geojson_to_coco.py
import json, argparse
from pathlib import Path

def ring_bbox(r):
    xs=[float(x) for x,_ in r]; ys=[float(y) for _,y in r]
    x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
    return [x1,y1,max(0.0,x2-x1),max(0.0,y2-y1)]

def image_key(meta):
    src=(meta or {}).get("source",{})
    img=(src or {}).get("image",{})
    return img.get("id") or src.get("filename") or "unknown"

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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_dir", default="out_pred")
    ap.add_argument("--class_key", choices=["classification","label","auto"], default="auto")
    args=ap.parse_args()

    fc=json.load(open(args.in_path))
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    feats=fc.get("features",[])
    file_meta=fc.get("metadata",{})

    cat2id={}; next_cid=1
    img2id={}; next_img=1
    preds=[]

    def get_cls(props):
        if args.class_key=="classification": return props.get("classification")
        if args.class_key=="label": return props.get("label")
        return props.get("classification") if props.get("classification") is not None else props.get("label")

    default_key=image_key(file_meta)
    if default_key not in img2id: img2id[default_key]=next_img; next_img+=1

    for f in feats:
        props=f.get("properties",{}); geom=f.get("geometry",{}); meta=f.get("metadata") or file_meta
        ik=image_key(meta) or default_key
        if ik not in img2id: img2id[ik]=next_img; next_img+=1
        img_id=img2id[ik]

        cls=get_cls(props)
        if cls not in cat2id: cat2id[cls]=next_cid; next_cid+=1
        cid=cat2id[cls]

        r=first_ring(geom)
        if not r or len(r)<3: continue
        x,y,w,h=ring_bbox(r)
        if w<=0 or h<=0: continue

        score=float(props.get("confidence",1.0))
        preds.append({"image_id":img_id,"category_id":cid,"bbox":[x,y,w,h],"score":score})

    json.dump(preds, open(out_dir/"preds.json","w"), indent=2)
    json.dump({"image_id_map": img2id, "category_id_map": {str(k):v for k,v in cat2id.items()}},
              open(out_dir/"mappings.json","w"), indent=2)
    print(f"[preds→coco] detections:{len(preds)} → {out_dir}/preds.json; maps → {out_dir}/mappings.json")

if __name__=="__main__":
    main()
