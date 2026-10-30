"""Build the 12 census-neighbour features used by the national model."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *
F = pd.read_parquet(D / "frame_systems.parquet")
F = F[F.lat.notna()].reset_index(drop=True)
F["sw"] = F.GW_SW_CODE.eq("SW").astype(int); F["purch"] = F.PRIMARY_SOURCE_CODE.fillna("").str.endswith("P").astype(int)
F["cws"] = F.PWS_TYPE_CODE.eq("CWS").astype(int); F["owner"] = F.OWNER_TYPE_CODE.astype("category").cat.codes
F["school"] = F.IS_SCHOOL_OR_DAYCARE_IND.eq("Y").astype(int); F["whole"] = F.IS_WHOLESALER_IND.eq("Y").astype(int)
F["lpop"] = np.log10(F["pop"].clip(lower=1)); F["lconn"] = np.log10(F.conn.clip(lower=1))
F["state"] = F.PRIMACY_AGENCY_CODE.astype("category").cat.codes; F["region"] = pd.to_numeric(F.EPA_REGION, errors="coerce")
F["age"] = 2026 - pd.to_datetime(F.FIRST_REPORTED_DATE, format="%m/%d/%Y", errors="coerce").dt.year
# ---- neighbour labels from census-monitored (medium/large) systems; exclude self
ref = F[F.size_cat.isin(["medium", "large"]) & F.y_mcl.notna()]
tree = BallTree(np.radians(ref[["lat", "lon"]].values), metric="haversine")
R_EARTH = 6371.0
idx, dist = tree.query_radius(np.radians(F[["lat", "lon"]].values), r=50 / R_EARTH, return_distance=True)
refy = ref.y_mcl.values; refd = ref.y_det.values; refid = ref.PWSID.values
feat = {k: np.full(len(F), np.nan) for k in ["nb_n10", "nb_mcl10", "nb_det10", "nb_n25", "nb_mcl25", "nb_det25", "nb_n50", "nb_mcl50", "nb_det50", "nb_kmcl", "nb_kdet", "nb_dmin"]}
for i, (ii, dd) in enumerate(zip(idx, dist)):
    keep = refid[ii] != F.PWSID.iat[i]; ii = ii[keep]; dd = dd[keep] * R_EARTH
    for r in [10, 25, 50]:
        m = dd <= r; feat[f"nb_n{r}"][i] = m.sum()
        if m.sum(): feat[f"nb_mcl{r}"][i] = refy[ii[m]].mean(); feat[f"nb_det{r}"][i] = refd[ii[m]].mean()
    if len(dd):
        w = np.exp(-dd / 10.0); feat["nb_kmcl"][i] = (w * refy[ii]).sum() / w.sum(); feat["nb_kdet"][i] = (w * refd[ii]).sum() / w.sum(); feat["nb_dmin"][i] = dd.min()
for k, v in feat.items(): F[k] = v
F.to_parquet(D / "frame_features.parquet", index=False)
