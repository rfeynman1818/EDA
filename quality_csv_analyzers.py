# A. Grainy/low-contrast (background noise dominates)

import pandas as pd, numpy as np
df = pd.read_csv("ranked_quality.csv")
cand = df[df.bad_flag] if (df.bad_flag==True).any() else df
out = cand.sort_values(["dr_db","peak_db","score"], ascending=[True,True,True])
print(out[["rank","file","dr_db","peak_db","cfar","noise_proxy","score"]].head(25).to_string(index=False))

# B. Bright speckle / hot clutter (noisy tails)

import pandas as pd, numpy as np
df = pd.read_csv("ranked_quality.csv")
out = df.sort_values(["noise_proxy","cfar","peak_db"], ascending=[False,False,True])
print(out[["rank","file","noise_proxy","cfar","dr_db","peak_db","score"]].head(25).to_string(index=False))

# robust “noise index”
# Normalize columns and combine:

import pandas as pd, numpy as np
df = pd.read_csv("ranked_quality.csv")
for c in ["dr_db","peak_db","cfar","noise_proxy"]:
    df[c+"_z"] = (df[c]-df[c].mean())/(df[c].std() + 1e-9)
# higher index ⇒ “noisier”
df["noise_index"] = (-df["dr_db_z"] - df["peak_db_z"] - np.log10(df["cfar"]+1).replace([-np.inf,np.inf],0).fillna(0) + df["noise_proxy_z"])
out = df.sort_values("noise_index", ascending=False)
print(out[["file","noise_index","dr_db","peak_db","cfar","noise_proxy"]].head(25).to_string(index=False))
