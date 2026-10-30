"""Full small-system frame (all 56,591 active small CWS/NTNCWS, incl. 2,780 without coordinates) with a census-anchored
prior, and the national small-system PFOA/PFOS burden re-estimated with Korn-Graubard-type (KG) intervals.

Prior: 5-seed LightGBM ensemble trained on the medium+large census only (SDWIS + geography + census neighbour labels;
missing coordinates/neighbour features are left as NaN and handled by the booster). Census out-of-fold predictions give
an isotonic calibration map (pc, used by the classical Neyman planner) and an error model u = E|y - f| (used by ANCHOR).
Burden: the UCMR5 small-system sample is treated as stratified SRS within reconstructed cells (source type x 4 size
classes); estimators are the post-stratified Hajek estimator and the stratified PPI++ difference estimator.
Outputs: data_derived/small_frame_anchor.parquet, results/small_burden_kg.csv
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from common import *
import anchor_core as ac

BASE = ["lat", "lon", "sw", "purch", "cws", "owner", "school", "whole", "lpop", "lconn", "state", "region", "age",
        "n_wells", "n_intakes", "n_springs", "n_consec", "n_sources", "n_treat_plants", "n_sellers"]
GB = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, verbose=-1, n_jobs=8)


def build_table():
    """Derived columns computed once over the whole frame (consistent category codes); census neighbour labels merged."""
    F = pd.read_parquet(D / "frame_systems.parquet")
    nb = pd.read_parquet(D / "frame_features.parquet")
    NB = [c for c in nb.columns if c.startswith("nb_")]
    F = F.merge(nb[["PWSID"] + NB], on="PWSID", how="left")
    F["sw"] = F.GW_SW_CODE.eq("SW").astype(int); F["purch"] = F.PRIMARY_SOURCE_CODE.fillna("").str.endswith("P").astype(int)
    F["cws"] = F.PWS_TYPE_CODE.eq("CWS").astype(int); F["owner"] = F.OWNER_TYPE_CODE.astype("category").cat.codes
    F["school"] = F.IS_SCHOOL_OR_DAYCARE_IND.eq("Y").astype(int); F["whole"] = F.IS_WHOLESALER_IND.eq("Y").astype(int)
    F["lpop"] = np.log10(F["pop"].clip(lower=1)); F["lconn"] = np.log10(F.conn.clip(lower=1))
    F["state"] = F.PRIMACY_AGENCY_CODE.astype("category").cat.codes; F["region"] = pd.to_numeric(F.EPA_REGION, errors="coerce")
    F["age"] = 2026 - pd.to_datetime(F.FIRST_REPORTED_DATE, format="%m/%d/%Y", errors="coerce").dt.year
    return F, BASE + NB


def census_prior(F, X, seeds=(0, 1, 2, 3, 4)):
    cen = F[F.size_cat.isin(["medium", "large"]) & F.y_mcl.notna()].reset_index(drop=True)
    sm = F[F.size_cat == "small"].reset_index(drop=True)
    y = cen.y_mcl.values.astype(int); f = np.zeros(len(sm)); u = np.zeros(len(sm)); oof = np.zeros(len(cen))
    for s in seeds:
        f += lgb.LGBMClassifier(**GB, random_state=s).fit(cen[X], y).predict_proba(sm[X])[:, 1] / len(seeds)
        o = np.zeros(len(cen))
        for a, b in KFold(5, shuffle=True, random_state=s).split(cen):
            o[b] = lgb.LGBMClassifier(**GB, random_state=s).fit(cen[X].iloc[a], y[a]).predict_proba(cen[X].iloc[b])[:, 1]
        oof += o / len(seeds)
        u += np.clip(lgb.LGBMRegressor(**GB, random_state=s).fit(cen[X], np.abs(y - o)).predict(sm[X]), 1e-3, None) / len(seeds)
    sm["f"] = f; sm["u"] = u
    sm["pc"] = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=0.999).fit(oof, y).predict(f)
    sm["src"] = np.where(sm.sw == 1, "SW", "GW")
    sm["scls"] = pd.cut(sm["pop"], [-1, 100, 500, 1000, 3300], labels=["<=100", "101-500", "501-1000", "1001-3300"]).astype(str)
    sm["cell"] = sm.src + "|" + sm.scls
    sm["P"] = sm["pop"].fillna(sm["pop"].median()).clip(lower=25)
    return sm


def stratified_burden(sm, estimand, use_pred):
    """Post-stratified estimate over the frame `sm` (cells as strata, SRS within cells); returns theta, se, df, n."""
    cells = sm.cell.astype("category").cat.codes.values; H = cells.max() + 1
    lab = sm.y_mcl.notna().values
    Nh = np.bincount(cells, minlength=H).astype(float); nh = np.bincount(cells[lab], minlength=H)
    keep = nh >= 2                                   # cells with <2 sampled units cannot carry a variance; report their frame share
    a = np.ones(len(sm)) if estimand == "systems" else sm.P.values
    inc = keep[cells]
    pi = np.where(lab, (nh / Nh)[cells], np.nan)
    m = lab & inc; A = a[inc].sum(); des = ("strat", cells[m], Nh, nh)
    y = sm.y_mcl.values[m].astype(float); w = 1 / pi[m]
    if use_pred:
        th, se, df = ac.est_ppi(y, sm.f.values[m], a[m], w, pi[m], A, np.sum(a[inc] * sm.f.values[inc]), des)
    else:
        th, se, df = ac.est_hajek(y, a[m], w, pi[m], A, des)
    return th, se, df, m.sum(), 1 - inc.mean()


if __name__ == "__main__":
    F, X = build_table()
    sm = census_prior(F, X)
    sm.drop(columns=[c for c in sm.columns if c.startswith("nb_")]).to_parquet(D / "small_frame_anchor.parquet", index=False)
    lab = sm[sm.y_mcl.notna()]
    print(f"small frame N={len(sm)} (no coordinates {sm.lat.isna().sum()}), sampled n={len(lab)}; prior AUC on sample {roc_auc_score(lab.y_mcl, lab.f):.3f}")
    rows = []
    groups = [("all", sm)] + [(f"size {k}", g) for k, g in sm.groupby("scls")] + [(f"source {k}", g) for k, g in sm.groupby("src")]
    for gname, g in groups:
        for estimand in ["systems", "people"]:
            tot = len(g) if estimand == "systems" else g.P.sum()
            for est, use in [("design-only (post-stratified Hajek)", False), ("ANCHOR PPI++", True)]:
                th, se, df, n, excl = stratified_burden(g, estimand, use)
                iv = ac.intervals_vec([th], [se], [df], [n])
                lo, hi = iv["kg"][0][0], iv["kg"][1][0]; wl, wh = iv["wald"][0][0], iv["wald"][1][0]
                rows.append(dict(group=gname, estimand=estimand, estimator=est, n=n, N=len(g), theta=th, se=se, kg_lo=lo, kg_hi=hi,
                                 wald_lo=wl, wald_hi=wh, count=th * tot, count_lo=lo * tot, count_hi=hi * tot, excluded_share=excl))
            a = np.ones(len(g)) if estimand == "systems" else g.P.values
            rows.append(dict(group=gname, estimand=estimand, estimator="ML plug-in (census model, no rectification)", n=0, N=len(g),
                             theta=np.sum(a * g.f) / a.sum(), count=np.sum(a * g.f) / a.sum() * tot))
    out = pd.DataFrame(rows); out.to_csv(R / "small_burden_kg.csv", index=False)
    pd.set_option("display.width", 220)
    show = out.assign(theta=out.theta.round(4), kg=out.apply(lambda r: f"[{r.kg_lo:.4f}, {r.kg_hi:.4f}]" if r.n else "", axis=1),
                      count=out["count"].round(0), cnt=out.apply(lambda r: f"[{r.count_lo:,.0f}, {r.count_hi:,.0f}]" if r.n else "", axis=1))
    print(show[["group", "estimand", "estimator", "n", "N", "theta", "kg", "count", "cnt", "excluded_share"]].to_string(index=False))
