"""Build location-event (wide, censored) and location-level tables from the official UCMR5 final release.

Unit of a row in `le`: PWSID x FacilityID x SamplePointID x SampleEventCode.
Values are ng/L; `<` results are kept as left-censored at the MRL (never set to 0 or MRL/2 here).
"""
import pathlib, numpy as np, pandas as pd
from project_paths import D
a = pd.read_parquet(D / "ucmr5_all.parquet")
a["val"] = pd.to_numeric(a.AnalyticalResultValue, errors="coerce") * 1000
a["mrl"] = pd.to_numeric(a.MRL, errors="coerce") * 1000
a["date"] = pd.to_datetime(a.CollectionDate, format="%m/%d/%Y", errors="coerce")
loc = ["PWSID", "FacilityID", "SamplePointID"]
ev = loc + ["SampleEventCode"]
# duplicates of the same analyte within one event
dup = a.groupby(ev + ["Contaminant"]).size()
print("analyte-event duplicates:", (dup > 1).sum(), "of", len(dup))
# keep detected value if any; otherwise the (max) censoring point
a["det"] = a.val.notna()
agg = a.groupby(ev + ["Contaminant"]).agg(val=("val", "max"), mrl=("mrl", "max"), det=("det", "max"),
                                           date=("date", "min"), method=("MethodID", "first")).reset_index()
V = agg.pivot_table(index=ev, columns="Contaminant", values="val", aggfunc="first")
M = agg.pivot_table(index=ev, columns="Contaminant", values="mrl", aggfunc="first")
meta = a.groupby(ev).agg(date=("date", "min"), date_max=("date", "max"), Size=("Size", "first"),
                         WaterType=("FacilityWaterType", "first"), State=("State", "first"),
                         Region=("Region", "first"), n_methods=("MethodID", "nunique"))
V.columns = ["v_" + c for c in V.columns]; M.columns = ["m_" + c for c in M.columns]
le = meta.join(V).join(M).reset_index()
le.to_parquet(D / "loc_event.parquet", index=False)
print("location-events:", le.shape, "locations:", le[loc].drop_duplicates().shape[0])
print(le.groupby("WaterType").SampleEventCode.value_counts().unstack().to_string())
n_ev = le.groupby(loc).SampleEventCode.nunique()
print("events per location:", n_ev.value_counts().sort_index().to_dict())
print("date span within event (days) >0:", ((le.date_max - le.date).dt.days > 0).mean())
