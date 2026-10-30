"""Within-system sampling depth on real UCMR5 schedules: analytical samples used and system misclassification per rule.

Unit: system (PWS) with every location on a complete schedule (SW/GU/MX: 4 quarterly events; GW: 2 semi-annual events).
Truth y: any location with a PFOA or PFOS average >= 4.05 ng/L (<MRL counted as 0).
Per-location rules (a stopped location is classified as not exceeding; a continued location completes its schedule):
  full : every scheduled event
  R1   : event 1; stop if PFOA and PFOS are both below the MRL
  R2   : events 1-2; stop if both events are below the MRL for PFOA and PFOS (GW locations: equals full)
  R12  : R1 at GW locations, R2 at surface-water locations
System classification y_tilde = any continued location exceeding; cost = events taken over all locations.
Output: results/within_system_rules.parquet (one row per system, all size classes).
"""
import numpy as np, pandas as pd
from common import *

le = complete_schedule()
le["det"] = le[["v_PFOA", "v_PFOS"]].notna().any(axis=1)
le["pf"] = le.v_PFOA.fillna(0); le["ps"] = le.v_PFOS.fillna(0)
loc = le.groupby(LOC).agg(n_ev=("k", "size"), sw=("sw", "first"), avg_a=("pf", "mean"), avg_s=("ps", "mean")).reset_index()
d = le.pivot_table(index=LOC, columns="k", values="det", aggfunc="first").reindex(loc.set_index(LOC).index)
loc["y"] = ((loc.avg_a >= MCL["PFOA"]) | (loc.avg_s >= MCL["PFOS"])).astype(int)
d1 = d[1].astype("boolean").fillna(False).to_numpy(bool); d2 = d[2].astype("boolean").fillna(False).to_numpy(bool)
n = loc.n_ev.values; y = loc.y.values; sw = loc.sw.values
stop1 = ~d1; stop2 = (~d1) & (~d2) & (n > 2)
rules = {"full": (n, y), "R1": (np.where(stop1, 1, n), np.where(stop1, 0, y)),
         "R2": (np.where(stop2, 2, n), np.where(stop2, 0, y))}
rules["R12"] = (np.where(sw, rules["R2"][0], rules["R1"][0]), np.where(sw, rules["R2"][1], rules["R1"][1]))
for r, (c, yt) in rules.items():
    loc[f"cost_{r}"] = c; loc[f"yt_{r}"] = yt
S = loc.groupby("PWSID").agg(y=("y", "max"), n_loc=("y", "size"), any_sw=("sw", "max"),
                             **{f"cost_{r}": (f"cost_{r}", "sum") for r in rules}, **{f"yt_{r}": (f"yt_{r}", "max") for r in rules}).reset_index()
F = pd.read_parquet(D / "frame_systems.parquet")[["PWSID", "size_cat", "pop"]]
S = S.merge(F, on="PWSID", how="left")
S.to_parquet(R / "within_system_rules.parquet", index=False)
rows = []
for grp, g in [("small (UCMR5 sample)", S[S.size_cat == "small"]), ("medium census", S[S.size_cat == "medium"]), ("large census", S[S.size_cat == "large"])]:
    for r in rules:
        fn = ((g[f"yt_{r}"] == 0) & (g.y == 1)).sum()
        rows.append(dict(group=grp, rule=r, systems=len(g), mean_cost=g[f"cost_{r}"].mean(), cost_saved=1 - g[f"cost_{r}"].sum() / g.cost_full.sum(),
                         exceeders=int(g.y.sum()), missed=int(fn), miss_rate=fn / max(g.y.sum(), 1), prev_full=g.y.mean(), prev_trunc=g[f"yt_{r}"].mean()))
out = pd.DataFrame(rows); out.to_csv(R / "within_system_rules_summary.csv", index=False)
print(out.round(3).to_string(index=False))
