# analyze.py
import json, csv, argparse, collections
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

def iou_xywh(a, b):
    ax1, ay1, aw, ah = map(float, a); bx1, by1, bw, bh = map(float, b)
    ax2, ay2, bx2, by2 = ax1+aw, ay1+ah, bx1+bw, by1+bh
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    a_area, b_area = max(0.0,aw)*max(0.0,ah), max(0.0,bw)*max(0.0,bh)
    denom = a_area + b_area - inter
    return inter/denom if denom>0 else 0.0

def draw_box(draw, bbox, color, width=3):
    x,y,w,h = bbox
    for k in range(width):
        draw.rectangle([x-k,y-k,x+w+k,y+h+k], outline=color)

def safe_font(size=14):
    try: return ImageFont.truetype("arial.ttf", size)
    except: return ImageFont.load_default()

def match_per_image(gt_objs, pred_objs, iou_thr):
    gt_used=set(); matches=[]
    pred_sorted = sorted(pred_objs, key=lambda p: (-p["score"], p["category_id"]))
    for p in pred_sorted:
        best_iou, best_j = -1.0, -1
        for j,g in enumerate(gt_objs):
            if j in gt_used or g["category_id"]!=p["category_id"]: continue
            i = iou_xywh(p["bbox"], g["bbox"])
            if i>=iou_thr and i>best_iou:
                best_iou, best_j = i, j
        if best_j>=0:
            gt_used.add(best_j)
            matches.append((p, gt_objs[best_j], best_iou))
    tp=len(matches); fp=max(0,len(pred_objs)-tp); fn=max(0,len(gt_objs)-tp)
    return matches, tp, fp, fn

def prf(tp, fp, fn):
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return prec, rec, f1

def summarize(args):
    with open(args.gt_json) as f: gt = json.load(f)
    with open(args.pred_json) as f: preds = json.load(f)

    img_by_id = {im["id"]: im for im in gt.get("images", [])}
    cats = {c["id"]: c["name"] for c in gt.get("categories", [])}
    cname = lambda cid: cats.get(cid, str(cid))

    gts_by_img = collections.defaultdict(list)
    for a in gt.get("annotations", []):
        if a.get("iscrowd",0)==1: continue
        gts_by_img[a["image_id"]].append({"bbox":a["bbox"], "category_id":a["category_id"]})

    preds_by_img = collections.defaultdict(list)
    for p in preds:
        if p.get("score",0.0) < args.score_thr: continue
        preds_by_img[p["image_id"]].append({"bbox":p["bbox"], "category_id":p["category_id"], "score":p["score"]})

    image_ids = sorted(set(img_by_id.keys()) | set(gts_by_img.keys()) | set(preds_by_img.keys()))
    per_image=[]
    class_fp=collections.Counter(); class_fn=collections.Counter()
    per_class_tp=collections.Counter(); per_class_pred=collections.Counter(); per_class_gt=collections.Counter()
    conf_fp=collections.Counter(); conf_fn=collections.Counter()

    for image_id in image_ids:
        gt_objs=gts_by_img.get(image_id, [])
        pred_objs=preds_by_img.get(image_id, [])
        matches, tp, fp, fn = match_per_image(gt_objs, pred_objs, args.iou_thr)

        matched_gt_idx=set(); matched_pred_idx=set()
        gt_idx={id(g):i for i,g in enumerate(gt_objs)}
        pr_idx={id(p):i for i,p in enumerate(pred_objs)}
        for p,g,_ in matches:
            matched_gt_idx.add(gt_idx[id(g)])
            matched_pred_idx.add(pr_idx[id(p)])
            per_class_tp[cname(g["category_id"])]+=1

        for i,p in enumerate(pred_objs):
            per_class_pred[cname(p["category_id"])]+=1
            if i not in matched_pred_idx:
                class_fp[cname(p["category_id"])] += 1
                conf_fp[(cname(p["category_id"]), "BG")] += 1

        for i,g in enumerate(gt_objs):
            per_class_gt[cname(g["category_id"])] += 1
            if i not in matched_gt_idx:
                class_fn[cname(g["category_id"])] += 1
                conf_fn[("BG", cname(g["category_id"]))] += 1

        prec, rec, f1 = prf(tp, fp, fn)
        per_image.append({
            "image_id": image_id,
            "file_name": img_by_id.get(image_id, {}).get("file_name", str(image_id)),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4),
            "num_gt": len(gt_objs), "num_pred": len(pred_objs)
        })

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fields=["image_id","file_name","tp","fp","fn","precision","recall","f1","num_gt","num_pred"]
    with open(out_dir/"per_image_metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_image)

    pool=[r for r in per_image if (r["num_gt"]>0 or r["num_pred"]>0)]
    if args.rank_by=="f1":
        ranked = sorted(pool, key=lambda r: (r["f1"], -(r["fp"]+r["fn"]), -r["num_gt"], r["file_name"]))
    elif args.rank_by=="errors":
        ranked = sorted(pool, key=lambda r: (-(r["fp"]+r["fn"]), r["f1"], r["file_name"]))
    else:
        ranked = sorted(pool, key=lambda r: (r["f1"], -(r["fp"]+r["fn"]), r["file_name"]))
    with open(out_dir/"worst_images.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(ranked[:args.top_k])

    total_tp=sum(r["tp"] for r in per_image)
    total_fp=sum(r["fp"] for r in per_image)
    total_fn=sum(r["fn"] for r in per_image)
    g_prec,g_rec,g_f1 = prf(total_tp,total_fp,total_fn)

    rows=[]
    for c in sorted(set(list(per_class_gt.keys())+list(per_class_pred.keys()))):
        tp=per_class_tp[c]; fp=class_fp[c]; fn=class_fn[c]
        pc,rc,f1=prf(tp,fp,fn)
        rows.append({"class":c,"tp":tp,"fp":fp,"fn":fn,"precision":round(pc,4),
                     "recall":round(rc,4),"f1":round(f1,4),"gt":per_class_gt[c],"pred":per_class_pred[c]})
    with open(out_dir/"per_class_metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                         ["class","tp","fp","fn","precision","recall","f1","gt","pred"])
        w.writeheader(); w.writerows(rows)

    obs={"total_images":len(per_image),
         "total_gt":sum(r["num_gt"] for r in per_image),
         "total_preds":sum(r["num_pred"] for r in per_image),
         "total_tp":total_tp,"total_fp":total_fp,"total_fn":total_fn,
         "global_precision":round(g_prec,4),"global_recall":round(g_rec,4),"global_f1":round(g_f1,4),
         "fp_by_class":collections.Counter(class_fp).most_common(),
         "fn_by_class":collections.Counter(class_fn).most_common()}
    json.dump(obs, open(out_dir/"observations.json","w"), indent=2)

    if args.confusion_csv:
        rows = [{"section":"FP","pred_class":k[0],"gt_class":k[1],"count":v} for k,v in conf_fp.items()]
        rows+= [{"section":"FN","pred_class":k[0],"gt_class":k[1],"count":v} for k,v in conf_fn.items()]
        with open(out_dir/args.confusion_csv,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=["section","pred_class","gt_class","count"])
            w.writeheader(); w.writerows(rows)

    if args.render_dir:
        rdir=Path(args.render_dir); rdir.mkdir(parents=True, exist_ok=True)
        font=safe_font(14); palette={"TP":"green","FP":"red","FN":"orange"}
        for r in ranked[:args.top_k]:
            image_id=r["image_id"]
            gt_objs=gts_by_img.get(image_id, [])
            pred_objs=preds_by_img.get(image_id, [])
            matches,_,_,_ = match_per_image(gt_objs, pred_objs, args.iou_thr)
            matched_gt=set(id(g) for _,g,_ in matches)
            matched_pred=set(id(p) for p,_,_ in matches)

            file_name = (img_by_id.get(image_id, {}) or {}).get("file_name")
            if not file_name: continue
            img_path=Path(args.images_root)/file_name
            if not img_path.exists(): continue

            im=Image.open(img_path).convert("RGB"); draw=ImageDraw.Draw(im)
            for p in pred_objs:
                tag="TP" if id(p) in matched_pred else "FP"
                draw_box(draw, p["bbox"], palette[tag], width=3)
                x,y,w,h=p["bbox"]
                draw.text((x, max(0,y-16)), f"{tag} {cname(p['category_id'])} {p['score']:.2f}",
                          fill=palette[tag], font=font)
            for g in gt_objs:
                if id(g) not in matched_gt:
                    draw_box(draw, g["bbox"], palette["FN"], width=2)
                    x,y,w,h=g["bbox"]
                    draw.text((x, max(0,y-16)), f"FN {cname(g['category_id'])}",
                              fill=palette["FN"], font=font)
            out_name = f"{int(image_id):012d}_overlay.png" if isinstance(image_id,int) else f"{str(image_id)}_overlay.png"
            im.save(rdir/out_name)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--images_root", default="")
    ap.add_argument("--out_dir", default="analysis_out")
    ap.add_argument("--render_dir", default="")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.05)
    ap.add_argument("--rank_by", choices=["f1","errors"], default="f1")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--confusion_csv", default="")
    args=ap.parse_args()
    summarize(args)
