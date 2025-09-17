# viz_nitf_csv.py
import argparse, pandas as pd, numpy as np
import plotly.express as px, plotly.io as pio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="quality_dashboard.html")
    ap.add_argument("--label-all", action="store_true")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    cols = ["file","rank","score","dr_db","peak_db","cfar","noise_proxy","bad_flag","error"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["bad_flag"] = df["bad_flag"].astype(bool)
    df["has_error"] = df["error"].fillna("").ne("")
    df["basename"] = df["file"].astype(str).str.replace(r".*[\\/]", "", regex=True)

    mask_valid = ~df["score"].isna() & ~df["dr_db"].isna() & ~df["peak_db"].isna()
    d = df[mask_valid].copy()

    if args.label_all:
        d["label"] = d["basename"]
    else:
        top = d.nlargest(args.topk, "score").index
        low = d.nsmallest(args.topk, "score").index
        flagged = d.index[d["bad_flag"]]
        show_idx = set(top) | set(low) | set(flagged)
        d["label"] = np.where(d.index.isin(show_idx), d["basename"], "")

    common_hover = {"file": True, "rank": True, "score": ":.2f", "dr_db":":.2f", "peak_db":":.2f",
                    "cfar":":d", "noise_proxy":":.3g", "bad_flag":True}
    figs = []

    f1 = px.scatter(d, x="dr_db", y="peak_db", size="cfar", color="bad_flag",
                    hover_data=common_hover, text="label",
                    title="Dynamic Range vs Peak (size=CFAR, color=bad_flag)")
    f1.update_traces(textposition="top center")
    figs.append(f1)

    for x in ["dr_db","peak_db","cfar","noise_proxy","rank"]:
        if x in d.columns:
            fx = px.scatter(d, x=x, y="score", color="bad_flag",
                            hover_data=common_hover, text="label",
                            title=f"Score vs {x}")
            fx.update_traces(textposition="top center")
            figs.append(fx)

    fh = px.histogram(d, x="score", nbins=40, title="Score distribution")
    figs.append(fh)

    # simple correlations (printed to console)
    num = d[["dr_db","peak_db","cfar","noise_proxy","score"]].copy()
    print("Spearman correlations:\n", num.corr(method="spearman").round(3))
    print("\nTop by score:")
    print(d.sort_values("score", ascending=False)[["rank","score","basename"]].head(10).to_string(index=False))
    if d["bad_flag"].any():
        print("\nExamples flagged bad:")
        print(d[d["bad_flag"]][["rank","score","basename"]].head(10).to_string(index=False))

    pio.write_html(figs, file=args.out, include_plotlyjs="cdn", auto_open=False, full_html=True)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
