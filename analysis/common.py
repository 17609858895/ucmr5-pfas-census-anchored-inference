"""Shared loaders for the monitoring-design experiments."""
import pathlib, numpy as np, pandas as pd
from project_paths import PROJECT_ROOT as HERE, D, R, N_JOBS
LOC = ["PWSID", "FacilityID", "SamplePointID"]
PFAS25 = ["PFBA","PFPeA","PFHxA","PFHpA","PFOA","PFNA","PFDA","PFUnA","PFDoA","PFBS","PFPeS","PFHxS","PFHpS","PFOS",
          "4:2 FTS","6:2 FTS","8:2 FTS","HFPO-DA","ADONA","9Cl-PF3ONS","11Cl-PF3OUdS","PFMPA","PFMBA","PFEESA","NFDHA"]
MCL = {"PFOA": 4.05, "PFOS": 4.05}           # EPA significant-digit convention (>= 0.00405 ug/L)
MCL24 = {"PFOA": 4.05, "PFOS": 4.05, "PFHxS": 15, "PFNA": 15, "HFPO-DA": 15}
HBWC = {"PFHxS": 10, "PFNA": 10, "HFPO-DA": 10, "PFBS": 2000}

def safe(c):
    return c.replace(":", "_").replace(" ", "_").replace("-", "_")

def detected_cols(le):
    return [c for c in PFAS25 if f"v_{c}" in le.columns]

def complete_schedule():
    """Locations with a full UCMR5 schedule: 4 quarterly events (SW/GU/MX) or 2 semi-annual events (GW)."""
    le = pd.read_parquet(D / "loc_event.parquet")
    le["sw"] = le.WaterType.isin(["SW", "GU", "MX"])
    need = np.where(le.sw, 4, 2)
    n = le.groupby(LOC).SampleEventCode.transform("nunique")
    le = le[n.values == need].copy()
    le["k"] = le.SampleEventCode.str[-1].astype(int)
    return le.sort_values(LOC + ["k"])

def endpoints(le):
    """Analysis location averages (<MRL counted as 0, as in EPA's UCMR5-to-MCL comparison)."""
    g = le.groupby(LOC)
    out = pd.DataFrame({f"avg_{c}": g[f"v_{c}"].apply(lambda s: s.fillna(0).mean()) for c in set(MCL24) | set(HBWC)})
    out["E_PFOAS"] = ((out.avg_PFOA >= MCL["PFOA"]) | (out.avg_PFOS >= MCL["PFOS"])).astype(int)
    hi = sum(out[f"avg_{c}"] / HBWC[c] for c in HBWC)
    ndet_hi = g[[f"v_{c}" for c in HBWC]].apply(lambda d: int((d.notna().any()).sum()))
    out["HI"] = hi; out["E_HI"] = ((hi >= 1.5) & (ndet_hi >= 2)).astype(int)
    out["E_2024"] = (out.E_PFOAS | out.E_HI | (out.avg_PFHxS >= 15) | (out.avg_PFNA >= 15) | (out["avg_HFPO-DA"] >= 15)).astype(int)
    return out.reset_index()
