import os,socket,getpass
import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams["webagg.open_in_browser"]=False
mpl.rcParams["webagg.address"]=os.environ.get("WEBAGG_ADDR","127.0.0.1")
mpl.rcParams["webagg.port"]=int(os.environ.get("WEBAGG_PORT","8988"))
def _start_webagg():
    host=mpl.rcParams["webagg.address"]; port=mpl.rcParams["webagg.port"]; url=f"http://{host}:{port}/"
    try:
        from matplotlib.backends.backend_webagg import ServerThread
        srv=ServerThread(); srv.daemon=True; srv.start(); port=getattr(srv,"port",port); url=f"http://{host}:{port}/"
    except Exception: pass
    print("Open in browser:",url)
    who=getpass.getuser(); fqdn=socket.getfqdn() or "remote.host"
    print(f"If remote: ssh -N -L {port}:{host}:{port} {who}@{fqdn}\nThen open http://127.0.0.1:{port}/")
    return url
_=_start_webagg()
######################

import json,numpy as np,cv2
from pathlib import Path
import matplotlib.pyplot as plt,matplotlib.patches as patches
from matplotlib.lines import Line2D
from sarpy.io.complex.sicd import SICDReader
from sarpy.visualization.remap import Density

TARGET_SPACING=0.15
NITF_PATH=r"/path/to/your/file.nitf"
GEOJSON_PATH_1=r"/path/to/annotations1.geojson"
GEOJSON_PATH_2=r"/path/to/annotations2.geojson"
IOU_THRESHOLD=0.5
STYLES={"matched":{"color":"lime","linestyle":"-","label":"Matched"},
        "unmatched_file1":{"color":"aqua","linestyle":":","label":"Only in File 1"},
        "unmatched_file2":{"color":"yellow","linestyle":"--","label":"Only in File 2"}}

def iou(a,b):
    xl=max(a[0],b[0]); yt=max(a[1],b[1]); xr=min(a[2],b[2]); yb=min(a[3],b[3])
    if xr<=xl or yb<=yt: return 0.0
    inter=(xr-xl)*(yb-yt); ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1]); u=ua+ub-inter
    return 0.0 if u==0 else inter/u

def load_geojson(p):
    anns=[]
    with open(p,"r") as f: d=json.load(f)
    for feat in d.get("features",[]):
        props=feat.get("properties",{})
        coords=np.array(feat.get("image_coordinates",props.get("image_coordinates",[])))
        if coords.size==0: continue
        x_min,y_min=coords.min(axis=0); x_max,y_max=coords.max(axis=0)
        anns.append({"bbox":[x_min,y_min,x_max,y_max],"label":props.get("label","Unknown")})
    return anns

def compare(a1,a2,t):
    matched,un1,used=[],[],set()
    for x in a1:
        best,idx=0.0,-1
        for j,y in enumerate(a2):
            if j in used: continue
            v=iou(x["bbox"],y["bbox"])
            if v>best: best,idx=v,j
        if best>=t: matched.append({"anno1":x,"anno2":a2[idx]}); used.add(idx)
        else: un1.append(x)
    un2=[y for j,y in enumerate(a2) if j not in used]
    return {"matched":matched,"unmatched_file1":un1,"unmatched_file2":un2}

def _draw(ax,item,style,sx,sy):
    x0,y0,x1,y1=item["bbox"]; x=x0*sx; y=y0*sy; w=(x1-x0)*sx; h=(y1-y0)*sy
    ax.add_patch(patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor=style["color"],facecolor="none",linestyle=style["linestyle"]))
    ax.text(x,max(0,y-5),f'{style["label"]}: {item.get("label","")}',color=style["color"],fontsize=8,bbox=dict(facecolor="black",alpha=0.4,edgecolor="none"))

def draw_results(ax,res,styles,sx,sy):
    for p in res["matched"]: _draw(ax,p["anno1"],styles["matched"],sx,sy); _draw(ax,p["anno2"],styles["matched"],sx,sy)
    for it in res["unmatched_file1"]: _draw(ax,it,styles["unmatched_file1"],sx,sy)
    for it in res["unmatched_file2"]: _draw(ax,it,styles["unmatched_file2"],sx,sy)

def main():
    try:
        [plt.close() for _ in range(4)]
    except: pass
    print(f"Loading {Path(GEOJSON_PATH_1).name}"); a1=load_geojson(GEOJSON_PATH_1); print(len(a1))
    print(f"Loading {Path(GEOJSON_PATH_2).name}"); a2=load_geojson(GEOJSON_PATH_2); print(len(a2))
    print(f"Compare IoU≥{IOU_THRESHOLD}"); res=compare(a1,a2,IOU_THRESHOLD)
    print("Reading SICD"); r=SICDReader(NITF_PATH)
    rr=r.sicd_meta.Grid.Row.SS; rc=r.sicd_meta.Grid.Col.SS
    sy=rr/TARGET_SPACING; sx=rc/TARGET_SPACING
    H,W=r.sicd_meta.ImageData.NumRows,r.sicd_meta.ImageData.NumCols
    nh,nw=int(H*sy),int(W*sx)
    print(f"Remap+resize -> {nw}x{nh}")
    img=Density()(r[:]); img=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_CUBIC)
    fig,ax=plt.subplots(figsize=(15,15)); ax.imshow(img,cmap="gray")
    draw_results(ax,res,STYLES,sx,sy)
    handles=[Line2D([0],[0],color=v["color"],linestyle=v["linestyle"],linewidth=2,label=v["label"]) for v in STYLES.values()]
    ax.legend(handles=handles,loc="best",facecolor="lightgray",framealpha=0.7)
    ax.set_title(f"Comparison: {Path(GEOJSON_PATH_1).name} vs {Path(GEOJSON_PATH_2).name}\nImage: {Path(NITF_PATH).name} (TARGET_SPACING={TARGET_SPACING})")
    ax.axis("off"); plt.tight_layout(); print("Showing plot"); plt.show(block=False)

main()



