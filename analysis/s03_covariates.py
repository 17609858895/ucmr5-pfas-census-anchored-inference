"""Pre-sampling covariates per PWS (read-only from the user's existing raw downloads; nothing modified there).

SDWIS/SDWA public water system attributes (2026-10-14 download) and ZIP-centroid geography (2023 ZCTA gazetteer).
"""
import pathlib, zipfile, io, numpy as np, pandas as pd
from project_paths import D
from project_paths import SDWIS_ZIP, ZCTA_FILE
with zipfile.ZipFile(SDWIS_ZIP) as z, z.open("SDWA_PUB_WATER_SYSTEMS.csv") as f:
    pws = pd.read_csv(f, dtype=str, usecols=["PWSID", "PWS_TYPE_CODE", "OWNER_TYPE_CODE", "POPULATION_SERVED_COUNT",
                                             "PRIMARY_SOURCE_CODE", "IS_WHOLESALER_IND", "SERVICE_CONNECTIONS_COUNT",
                                             "FIRST_REPORTED_DATE"])
pws = pws.drop_duplicates("PWSID")
out = pd.DataFrame({"PWSID": pws.PWSID})
out["pop_log"] = np.log10(pd.to_numeric(pws.POPULATION_SERVED_COUNT, errors="coerce").clip(lower=1)).values
out["pop_served"] = pd.to_numeric(pws.POPULATION_SERVED_COUNT, errors="coerce").values
out["conn_log"] = np.log10(pd.to_numeric(pws.SERVICE_CONNECTIONS_COUNT, errors="coerce").clip(lower=1)).values
out["purchased"] = pws.PRIMARY_SOURCE_CODE.str.endswith("P").astype(float).values
out["wholesaler"] = (pws.IS_WHOLESALER_IND == "Y").astype(float).values
out["owner"] = pws.OWNER_TYPE_CODE.values
out["cws"] = (pws.PWS_TYPE_CODE == "CWS").astype(float).values
out["sys_age"] = (2023 - pd.to_datetime(pws.FIRST_REPORTED_DATE, format="%m/%d/%Y", errors="coerce").dt.year).values
zp = pd.read_parquet(D / "ucmr5_zip.parquet")
gz = pd.read_csv(ZCTA_FILE, sep="\t", dtype=str)
gz.columns = [c.strip() for c in gz.columns]
gz["lat"] = pd.to_numeric(gz.INTPTLAT); gz["lon"] = pd.to_numeric(gz.INTPTLONG); gz["aland"] = pd.to_numeric(gz.ALAND_SQMI)
zp = zp.merge(gz[["GEOID", "lat", "lon", "aland"]], left_on="ZIPCODE", right_on="GEOID", how="left")
geo = zp.groupby("PWSID").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_zip=("ZIPCODE", "nunique"),
                              zip_area_log=("aland", lambda s: np.log10(s.sum() + 1))).reset_index()
cov = geo.merge(out, on="PWSID", how="outer")
cov.to_parquet(D / "pws_covariates.parquet", index=False)
le = pd.read_parquet(D / "loc_event.parquet")
u = le.PWSID.drop_duplicates()
print("UCMR5 PWS with SDWA pop:", u.isin(out.PWSID[out.pop_served.notna()]).mean(), " with lat:", u.isin(geo.PWSID[geo.lat.notna()]).mean())
