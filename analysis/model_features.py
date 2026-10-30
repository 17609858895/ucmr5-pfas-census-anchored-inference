"""Cross-size feature definitions used by the published benchmark."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
BASE = ["lat", "lon", "sw", "purch", "cws", "owner", "school", "whole", "lpop", "lconn", "state", "region", "age",
        "n_wells", "n_intakes", "n_springs", "n_consec", "n_sources", "n_treat_plants", "n_sellers"]
Q = ["q_n50", "q_mcl50", "q_kmcl", "q_dmin", "q_mcl25"]
def nb_feats(target_df, ref_df):
    tree = BallTree(np.radians(ref_df[["lat", "lon"]].values), metric="haversine")
    idx, dist = tree.query_radius(np.radians(target_df[["lat", "lon"]].values), r=50 / 6371.0, return_distance=True)
    ry = ref_df.y_mcl.values; rid = ref_df.PWSID.values; out = np.full((len(target_df), 5), np.nan)
    for i, (ii, dd) in enumerate(zip(idx, dist)):
        k = rid[ii] != target_df.PWSID.iat[i]; ii, dd = ii[k], dd[k] * 6371.0
        if len(dd):
            w = np.exp(-dd / 10); out[i] = [len(dd), ry[ii].mean(), (w * ry[ii]).sum() / w.sum(), dd.min(), ry[ii][dd <= 25].mean() if (dd <= 25).any() else np.nan]
    return pd.DataFrame(out, columns=Q, index=target_df.index)
