# remap_to_gt.py
import json, argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds_raw", required=True)       # out_pred/preds.json
    ap.add_argument("--maps", required=True)            # out_pred/mappings.json
    ap.add_argument("--gt", required=True)              # coco_gt.json
    ap.add_argument("--out", default="out_pred/preds.remapped.json")
    ap.add_argument("--image_match_on", choices=["file_name","id"], default="file_name")
    args=ap.parse_args()

    preds = json.load(open(args.preds_raw))
    maps  = json.load(open(args.maps))
    gt    = json.load(open(args.gt))

    # local category id -> name
    name_to_local = {k:int(v) for k,v in maps["category_id_map"].items()}
    local_to_name = {v:k for k,v in name_to_local.items()}
    # GT name -> GT id
    gt_name_to_id = {c["name"]: int(c["id"]) for c in gt["categories"]}

    # local image id -> key
    key_to_local_img = {k:int(v) for k,v in maps["image_id_map"].items()}
    local_img_to_key = {v:k for k,v in key_to_local_img.items()}
    # key -> GT image id
    if args.image_match_on=="file_name":
        key_to_gt_img = {im["file_name"]: int(im["id"]) for im in gt["images"]}
    else:
        key_to_gt_img = {str(im["id"]): int(im["id"]) for im in gt["images"]}

    out=[]; miss_cat=miss_img=0
    for p in preds:
        cname = local_to_name.get(int(p["category_id"]))
        gt_cid = gt_name_to_id.get(cname)
        if gt_cid is None: miss_cat+=1; continue
        key = local_img_to_key.get(int(p["image_id"]))
        gt_imgid = key_to_gt_img.get(key)
        if gt_imgid is None: miss_img+=1; continue
        out.append({"image_id":gt_imgid,"category_id":gt_cid,"bbox":p["bbox"],"score":p.get("score",1.0)})

    json.dump(out, open(args.out,"w"), indent=2)
    print(f"wrote {len(out)} preds to {args.out} (dropped {miss_cat} unknown-class, {miss_img} unknown-image)")

if __name__=="__main__":
    main()

