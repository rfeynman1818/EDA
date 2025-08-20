# remap_to_gt.py
import json, argparse, os, sys

def load_gt_names_to_ids(gt, assume_numeric):
    if isinstance(gt, dict) and "categories" in gt:
        return {str(c["name"]): int(c["id"]) for c in gt["categories"]}
    if assume_numeric: return None
    raise SystemExit("[remap] GT file lacks 'categories'. Use --assume_numeric_labels if your class names are numeric IDs.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds_raw","--in", dest="preds_raw", required=True)
    ap.add_argument("--maps","--map", dest="maps", required=True)
    ap.add_argument("--gt", required=True, help="COCO GT from detection_matching_to_coco_gt.py")
    ap.add_argument("--out", default="out_pred/preds.remapped.json")
    ap.add_argument("--image_match_on", choices=["file_name","id"], default="file_name")
    ap.add_argument("--basename", action="store_true", help="When matching on file_name, compare basenames only")
    ap.add_argument("--assume_numeric_labels", action="store_true",
                    help="Treat class names as desired integer GT IDs (Path B typical)")
    args=ap.parse_args()

    preds=json.load(open(args.preds_raw))
    maps=json.load(open(args.maps))
    gt=json.load(open(args.gt))

    gt_name_to_id = load_gt_names_to_ids(gt, args.assume_numeric_labels)

    name_to_local = {k:int(v) for k,v in maps["category_id_map"].items()}
    local_to_name = {v:k for k,v in name_to_local.items()}

    if args.image_match_on=="file_name":
        def keyer(s): return os.path.basename(s) if args.basename else s
        key_to_gt_img = {keyer(im["file_name"]): int(im["id"]) for im in gt["images"]}
    else:
        key_to_gt_img = {str(im["id"]): int(im["id"]) for im in gt["images"]}

    key_to_local_img = {k:int(v) for k,v in maps["image_id_map"].items()}
    local_img_to_key = {v:k for k,v in key_to_local_img.items()}

    out=[]; miss_cat=miss_img=0
    for p in preds:
        if args.assume_numeric_labels:
            cname = local_to_name.get(int(p["category_id"]))
            try: gt_cid = int(str(cname))
            except: gt_cid = None
        else:
            cname = local_to_name.get(int(p["category_id"]))
            gt_cid = gt_name_to_id.get(str(cname))
        if gt_cid is None: miss_cat+=1; continue

        key = local_img_to_key.get(int(p["image_id"]))
        if args.image_match_on=="file_name" and args.basename: key = os.path.basename(key)
        gt_imgid = key_to_gt_img.get(str(key))
        if gt_imgid is None: miss_img+=1; continue

        out.append({"image_id": gt_imgid, "category_id": gt_cid, "bbox": p["bbox"], "score": p.get("score",1.0)})

    json.dump(out, open(args.out,"w"), indent=2)
    print(f"[remap] wrote {len(out)} preds → {args.out} (dropped {miss_cat} unknown-class, {miss_img} unknown-image)")

if __name__=="__main__":
    main()

