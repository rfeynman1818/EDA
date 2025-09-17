## Cell 1 — setup (renderer + imports)

# Notebook setup
import pandas as pd, numpy as np, os, io
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import ipywidgets as w
from IPython.display import display, clear_output

def setup_jupyter_interactivity():
    # Prefer a renderer that works in classic notebook and JupyterLab
    try:
        pio.renderers.default = 'notebook_connected'
    except Exception:
        pio.renderers.default = 'jupyterlab'  # fallback
    print("✅ Plotly interactive renderer set. If plots don't appear, try: pio.renderers.default='notebook_connected'")

## Cell 2 — CSV loader + utilities

METRIC_CHOICES = ["dr_db","peak_db","cfar","noise_proxy","score","rank"]

def _coerce_bool(x):
    if isinstance(x, bool): return x
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return s in ("1","true","t","y","yes")

def load_quality_csv(csv_path):
    dtypes = {"file":"string","rank":"Int64","score":"float","dr_db":"float","peak_db":"float","cfar":"Int64",
              "noise_proxy":"float","bad_flag":"string","error":"string"}
    df = pd.read_csv(csv_path, dtype=dtypes)
    if "bad_flag" in df.columns: df["bad_flag"] = df["bad_flag"].map(_coerce_bool)
    else: df["bad_flag"] = False
    df["error"] = df["error"].fillna("")
    df["basename"] = df["file"].astype(str).str.replace(r".*[\\/]", "", regex=True)
    # keep only rows with usable coords/scores
    mask = ~df["dr_db"].isna() & ~df["peak_db"].isna() & ~df["score"].isna()
    return df.loc[mask].reset_index(drop=True)

def filter_df(df, name_substr, bad_only, score_minmax, cfar_minmax):
    x = df
    if name_substr:
        s = name_substr.lower()
        x = x[x["basename"].str.lower().str.contains(s) | x["file"].str.lower().str.contains(s)]
    if bad_only: x = x[x["bad_flag"]]
    if score_minmax: x = x[(x["score"]>=score_minmax[0]) & (x["score"]<=score_minmax[1])]
    if cfar_minmax and "cfar" in x.columns and x["cfar"].notna().any():
        x = x[(x["cfar"]>=cfar_minmax[0]) & (x["cfar"]<=cfar_minmax[1])]
    return x

def choose_labels(df, label_all, topk):
    if label_all: return df["basename"].values
    top = set(df.nlargest(topk, "score").index)
    low = set(df.nsmallest(topk, "score").index)
    flagged = set(df.index[df["bad_flag"]])
    keep = top | low | flagged
    return np.where(df.index.isin(keep), df["basename"], "")


 ## Cell 3 — figure construction (all points labeled by default)

 def make_scatter_widget(df, x_key="dr_db", y_key="peak_db",
                        label_all=True, topk=12,
                        dr_th=None, peak_th=None):
    labels = choose_labels(df, label_all, topk)
    color = np.where(df["bad_flag"], "bad", "ok")
    size = df["cfar"].fillna(0).astype(float).clip(lower=0) + 1.0  # visible bubbles
    custom = np.stack([df["file"], df["rank"], df["score"], df["dr_db"], df["peak_db"], df["cfar"], df["noise_proxy"], df["bad_flag"]], axis=1)

    fig = go.FigureWidget(
        data=[go.Scatter(
            x=df[x_key], y=df[y_key],
            mode="markers+text",
            text=labels, textposition="top center",
            marker=dict(size=np.sqrt(size)*3, opacity=0.85, line=dict(width=0.5, color="black")),
            hovertemplate="<b>%{customdata[0]}</b><br>"+f"{x_key}=%{{x:.2f}}<br>{y_key}=%{{y:.2f}}<br>"+
                          "score=%{customdata[2]:.2f}<br>DR=%{customdata[3]:.2f} dB<br>Peak=%{customdata[4]:.2f} dB<br>"+
                          "CFAR=%{customdata[5]}<br>noise=%{customdata[6]:.3g}<br>bad=%{customdata[7]}<extra></extra>",
            customdata=custom,
            marker_color=np.where(df["bad_flag"], "#d62728", "#1f77b4"),
            name=f"{x_key} vs {y_key}"
        )]
    )
    fig.update_layout(height=650, title=f"{x_key} vs {y_key} (size≈CFAR, color=bad_flag; labels=filenames)",
                      xaxis_title=x_key, yaxis_title=y_key, margin=dict(l=60,r=20,b=60,t=60))
    # optional threshold lines
    shapes=[]
    if dr_th is not None and x_key=="dr_db":
        shapes.append(dict(type="line", x0=dr_th, x1=dr_th, y0=df[y_key].min(), y1=df[y_key].max(),
                           line=dict(dash="dash"), layer="below"))
    if peak_th is not None and y_key=="peak_db":
        shapes.append(dict(type="line", y0=peak_th, y1=peak_th, x0=df[x_key].min(), x1=df[x_key].max(),
                           line=dict(dash="dash"), layer="below"))
    if shapes: fig.update_layout(shapes=shapes)
    return fig

## Cell 4 — dashboard UI + callbacks (click to inspect; export filtered CSV)

def start_quality_dashboard(csv_path="ranked_quality.csv"):
    # widgets
    path_in = w.Text(value=csv_path, description="CSV:", layout=w.Layout(width="70%"))
    load_btn = w.Button(description="Load", button_style="primary")
    x_dd = w.Dropdown(options=METRIC_CHOICES, value="dr_db", description="X:")
    y_dd = w.Dropdown(options=METRIC_CHOICES, value="peak_db", description="Y:")
    label_all_cb = w.Checkbox(value=True, description="Show filename on ALL points")
    topk_sl = w.IntSlider(value=12, min=3, max=100, step=1, description="Label top/bot K", disabled=True)
    name_q = w.Text(value="", description="Filter name:")
    bad_only = w.Checkbox(value=False, description="Only bad_flag")
    score_rng = w.FloatRangeSlider(value=(float("-inf"), float("inf")), min=-1e6, max=1e6, step=0.1, description="Score range:")
    cfar_rng = w.IntRangeSlider(value=(0, 10_000), min=0, max=100_000, step=1, description="CFAR range:")
    dr_th_in = w.FloatText(value=None, description="DR thresh (x):")
    peak_th_in = w.FloatText(value=None, description="Peak thresh (y):")
    export_btn = w.Button(description="Export filtered CSV", button_style="")
    out = w.Output()
    details = w.Output()

    state = {"df": None, "df_filt": None, "fig": None}

    def refresh_ranges(df):
        with out:
            score_rng.min, score_rng.max = float(df["score"].min()), float(df["score"].max())
            score_rng.value = (score_rng.min, score_rng.max)
            if "cfar" in df.columns and df["cfar"].notna().any():
                cfar_rng.min, cfar_rng.max = int(df["cfar"].min()), int(df["cfar"].max())
                cfar_rng.value = (cfar_rng.min, cfar_rng.max)
            else:
                cfar_rng.min = cfar_rng.max = 0; cfar_rng.value = (0,0)

    def rebuild_figure(*_):
        if state["df"] is None: return
        df = filter_df(state["df"], name_q.value, bad_only.value, score_rng.value, cfar_rng.value)
        state["df_filt"] = df
        fig = make_scatter_widget(df, x_dd.value, y_dd.value,
                                  label_all_cb.value, topk_sl.value,
                                  dr_th=dr_th_in.value if dr_th_in.value==dr_th_in.value else None,
                                  peak_th=peak_th_in.value if peak_th_in.value==peak_th_in.value else None)
        # click callback
        def on_click(trace, points, selector):
            if not points.point_inds: return
            i = points.point_inds[0]
            row = df.iloc[i]
            with details:
                clear_output(wait=True)
                display(pd.DataFrame([{
                    "file": row["file"], "basename": row["basename"], "rank": row.get("rank"),
                    "score": row.get("score"), "dr_db": row.get("dr_db"), "peak_db": row.get("peak_db"),
                    "cfar": row.get("cfar"), "noise_proxy": row.get("noise_proxy"), "bad_flag": row.get("bad_flag"),
                    "error": row.get("error")
                }]))
        fig.data[0].on_click(on_click)
        state["fig"] = fig
        with out:
            clear_output(wait=True); display(fig)

    def on_load(_):
        try:
            df = load_quality_csv(path_in.value)
        except Exception as e:
            with out:
                clear_output(wait=True); print(f"❌ Failed to load CSV: {e}")
            return
        state["df"] = df
        refresh_ranges(df)
        rebuild_figure()

    def on_label_toggle(change):
        topk_sl.disabled = change["new"]
        rebuild_figure()

    def on_export(_):
        if state["df_filt"] is None or len(state["df_filt"])==0:
            with details:
                clear_output(wait=True); print("Nothing to export (filtered set is empty).")
            return
        buf = io.StringIO()
        state["df_filt"].to_csv(buf, index=False)
        b = w.Button(description="Download filtered.csv", icon="download")
        payload = buf.getvalue().encode()
        from base64 import b64encode
        href = f'<a download="filtered_ranked_quality.csv" href="data:text/csv;base64,{b64encode(payload).decode()}">Download filtered.csv</a>'
        with details:
            clear_output(wait=True); display(w.HTML(href))

    load_btn.on_click(on_load)
    label_all_cb.observe(on_label_toggle, names="value")
    for wid in (x_dd,y_dd,topk_sl,name_q,bad_only,score_rng,cfar_rng,dr_th_in,peak_th_in):
        wid.observe(lambda *_: rebuild_figure(), names="value")
    export_btn.on_click(on_export)

    controls_top = w.HBox([path_in, load_btn])
    controls_1 = w.HBox([x_dd, y_dd, label_all_cb, topk_sl])
    controls_2 = w.HBox([name_q, bad_only])
    controls_3 = w.HBox([score_rng, cfar_rng])
    controls_4 = w.HBox([dr_th_in, peak_th_in, export_btn])

    display(w.VBox([
        w.HTML("<b>QUALITY REVIEW DASHBOARD</b> — scatter with filename labels on all points"),
        controls_top, controls_1, controls_2, controls_3, controls_4,
        out, w.HTML("<hr><b>Point details (click a dot):</b>"), details
    ]))

    # auto-load if file exists
    if os.path.exists(csv_path): load_btn.click()

## Cell 5 — quick stats (optional)

def show_quality_stats(df):
    if df is None or len(df)==0: 
        print("No data."); return
    print(f"Rows: {len(df)} | Files: {df['file'].nunique()} | bad_flag: {df['bad_flag'].sum()}")
    print("\nSpearman correlations:")
    print(df[["dr_db","peak_db","cfar","noise_proxy","score"]].corr(method="spearman").round(3))

## Cell 6 — quick start

setup_jupyter_interactivity()
start_quality_dashboard("ranked_quality.csv")  # change path as needed
