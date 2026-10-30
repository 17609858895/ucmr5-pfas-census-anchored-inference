"""Sampling frame of all active CWS/NTNCWS (SDWIS 2026Q3) with covariates available for every system,
plus UCMR5 system-level PFAS outcomes for the monitored subset.

Location uses the administrative ZIP centroid for every system (same definition for monitored and unmonitored).
"""
import zipfile, numpy as np, pandas as pd
from common import *
from project_paths import SDWIS_ZIP, ZCTA_FILE
z = zipfile.ZipFile(SDWIS_ZIP)
with z.open("SDWA_PUB_WATER_SYSTEMS.csv") as f:
    p = pd.read_csv(f, dtype=str, usecols=["PWSID", "PWS_ACTIVITY_CODE", "PWS_TYPE_CODE", "GW_SW_CODE", "POPULATION_SERVED_COUNT",
                                         "SERVICE_CONNECTIONS_COUNT", "PRIMARY_SOURCE_CODE", "ZIP_CODE", "STATE_CODE", "PRIMACY_AGENCY_CODE",
                                         "OWNER_TYPE_CODE", "IS_WHOLESALER_IND", "IS_SCHOOL_OR_DAYCARE_IND", "FIRST_REPORTED_DATE", "EPA_REGION"])
p = p.drop_duplicates("PWSID")
p = p[(p.PWS_ACTIVITY_CODE == "A") & p.PWS_TYPE_CODE.isin(["CWS", "NTNCWS"])].copy()
p["pop"] = pd.to_numeric(p.POPULATION_SERVED_COUNT, errors="coerce")
p["conn"] = pd.to_numeric(p.SERVICE_CONNECTIONS_COUNT, errors="coerce")
p["size_cat"] = pd.cut(p["pop"], [-1, 3300, 10000, 1e10], labels=["small", "medium", "large"]).astype(str)
with z.open("SDWA_FACILITIES.csv") as f:
    fa = pd.read_csv(f, dtype=str, usecols=["PWSID", "FACILITY_ACTIVITY_CODE", "FACILITY_TYPE_CODE", "IS_SOURCE_IND", "WATER_TYPE_CODE", "SELLER_PWSID", "IS_SOURCE_TREATED_IND"])
fa = fa[(fa.FACILITY_ACTIVITY_CODE == "A") & fa.PWSID.isin(p.PWSID)]
src = fa[fa.IS_SOURCE_IND == "Y"]
fac = pd.DataFrame({"n_wells": src[src.FACILITY_TYPE_CODE == "WL"].groupby("PWSID").size(),
                    "n_intakes": src[src.FACILITY_TYPE_CODE == "IN"].groupby("PWSID").size(),
                    "n_springs": src[src.FACILITY_TYPE_CODE == "SP"].groupby("PWSID").size(),
                    "n_consec": src[src.FACILITY_TYPE_CODE == "CC"].groupby("PWSID").size(),
                    "n_sources": src.groupby("PWSID").size(),
                    "n_treat_plants": fa[fa.FACILITY_TYPE_CODE == "TP"].groupby("PWSID").size(),
                    "n_sellers": fa[fa.SELLER_PWSID.notna()].groupby("PWSID").SELLER_PWSID.nunique()})
p = p.merge(fac.fillna(0), left_on="PWSID", right_index=True, how="left")
gz = pd.read_csv(ZCTA_FILE, sep="\t", dtype=str)
gz.columns = [c.strip() for c in gz.columns]
gz = gz.assign(lat=pd.to_numeric(gz.INTPTLAT), lon=pd.to_numeric(gz.INTPTLONG))[["GEOID", "lat", "lon"]]
p["zip5"] = p.ZIP_CODE.str[:5]
p = p.merge(gz, left_on="zip5", right_on="GEOID", how="left").drop(columns="GEOID")
# fall back to the UCMR5-served-ZIP centroid where the admin ZIP is not a ZCTA (monitored systems only)
cov = pd.read_parquet(D / "pws_covariates.parquet")[["PWSID", "lat", "lon"]].rename(columns={"lat": "lat_u", "lon": "lon_u"})
p = p.merge(cov, on="PWSID", how="left")
p["geo_fallback"] = p.lat.isna() & p.lat_u.notna()
p["lat"] = p.lat.fillna(p.lat_u); p["lon"] = p.lon.fillna(p.lon_u); p = p.drop(columns=["lat_u", "lon_u"])
# ---- UCMR5 system outcomes (locations with a full schedule)
le = complete_schedule(); E = endpoints(le)
ys = E.groupby("PWSID").agg(y_mcl=("E_PFOAS", "max"), y_2024=("E_2024", "max"), n_loc=("E_PFOAS", "size")).reset_index()
det = le.groupby("PWSID")[[f"v_{c}" for c in detected_cols(le)]].count().sum(axis=1).gt(0).astype(int).rename("y_det").reset_index()
ys = ys.merge(det, on="PWSID", how="left")
allpws = le.PWSID.unique()
p["in_ucmr5"] = p.PWSID.isin(allpws)
p = p.merge(ys, on="PWSID", how="left")
p.to_parquet(D / "frame_systems.parquet", index=False)
print("frame", len(p), p.size_cat.value_counts().to_dict())
print("monitored by size:", p[p.in_ucmr5].size_cat.value_counts().to_dict(), " with outcome:", p[p.y_mcl.notna()].size_cat.value_counts().to_dict())
print("geo coverage", p.lat.notna().mean(), " fallback used", p.geo_fallback.sum())
print("small monitored: MCL prev %.4f det prev %.4f" % (p[(p.size_cat == "small") & p.y_mcl.notna()].y_mcl.mean(), p[(p.size_cat == "small") & p.y_det.notna()].y_det.mean()))
