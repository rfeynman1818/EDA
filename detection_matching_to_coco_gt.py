# detection_matching_to_coco_gt.py
import json, argparse
from pathlib import Path

def xyxy_to_xywh(b):
    x1,y1,x2,y2 = map(float, b)
    w=max(0.0,x2-x1); h=max(0.0,y2-y1)
    return [x1,y1,w,h], w*h

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default="coco_dm_gt.json")
    args=ap.parse_args()

    dm=json.load(open(args.in_path))
    pairs = dm["matched_detections"] if isinstance(dm,dict) and "matched_detections" in dm else dm

    img_key_to_id={}; next_img=1
    cat_ids=set()
    ann_seen=set()
    images=[]; annotations=[]; ann_id=1

    for e in pairs:
        ann=e.get("annotation",{})
        xyxy=ann.get("bbox"); label=ann.get("label"); img_key=str(ann.get("image",""))
        if not xyxy or label is None or img_key=="" or len(xyxy)!=4: continue

        if img_key not in img_key_to_id:
            img_key_to_id[img_key]=next_img
            images.append({"id": next_img, "file_name": img_key})
            next_img+=1
        img_id=img_key_to_id[img_key]

        key=(img_key, int(label), tuple(map(float,xyxy)))
        if key in ann_seen: continue
        ann_seen.add(key)

        bbox, area = xyxy_to_xywh(xyxy)
        annotations.append({"id":ann_id,"image_id":img_id,"category_id":int(label),
                            "bbox":bbox,"area":float(area),"iscrowd":0,"segmentation":[]})
        ann_id+=1; cat_ids.add(int(label))

    categories=[{"id":i,"name":str(i)} for i in sorted(cat_ids)]
    coco={"images":images,"annotations":annotations,"categories":categories}
    json.dump(coco, open(args.out_path,"w"), indent=2)
    print(f"[dm→coco-gt] images:{len(images)} anns:{len(annotations)} cats:{len(categories)} → {args.out_path}")

if __name__=="__main__":
    main()

