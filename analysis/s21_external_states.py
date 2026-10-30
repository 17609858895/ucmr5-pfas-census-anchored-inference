"""Independent external validation against state PFAS drinking-water data (Michigan, Minnesota, New Hampshire).

Downloaded 2026-10-25 into separate folders under external_raw/ (read-only here; run with `python -P` so neither the cwd nor a data folder is put on sys.path):
  MI  EGLE PublicWaterSupplySamplingResults (CWS + NTNCWS compliance samples 2020-08-18 .. 2025-12-24, RL 2 ng/L)
  MN  MDH Community Water Systems PFAS Sampling Results (per-system min/max, static table dated 2023-06-29; CWS only)
  NH  NHDES 'PWS PFAS Samples with 1000ft Buffer' (source-level flags vs state and federal MCLs)
Truth endpoints (closest available to the UCMR5 endpoint 'any location average >= 4.05 ng/L PFOA or PFOS'):
  MI  y_avg : any sampling location with mean(PFOA) or mean(PFOS) >= 4.05 ng/L, <RL counted as 0 (EPA convention)
      y_max : any single result of PFOA or PFOS >= 4 ng/L (sensitivity)
  MN  y_max : PFOA_Max or PFOS_Max >= 4 ng/L;  y_min : PFOA_Min or PFOS_Min >= 4 ng/L (sensitivity); purchased-water rows excluded
  NH  y_fed : any source flagged above the federal MCL (raw source water, not entry point)
Predictions are the census-anchored prior f from s19 (trained on UCMR5 medium/large systems only; no state data used).
Outputs: results/external_validation_systems.csv, results/external_validation_state_prevalence.csv, results/external_validation_report.md
"""
import sys, os, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -P drops the script directory from sys.path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *
import anchor_core as ac
from s19_small_frame_burden import stratified_burden

EXT = HERE / "external_raw"
rng = np.random.default_rng(2026)


def num(s):
    """'<2' / 'ND' / blank -> 0 (non-detect); numeric strings -> float."""
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.str.startswith("<") & ~s.str.upper().isin(["ND", "NAN", "", "NONE"]), "0"), errors="coerce").fillna(0.0)


def load_michigan():
    d = pd.read_csv(EXT / "michigan_egle_20261025/mi_pws_pfas_results.csv", dtype=str, encoding="utf-8-sig")
    d["PWSID"] = "MI" + d.WSSN.str.zfill(7); d["pfoa"] = num(d.PFOA); d["pfos"] = num(d.PFOS)
    loc = d.groupby(["PWSID", "LocName"]).agg(pfoa=("pfoa", "mean"), pfos=("pfos", "mean"), mx=("pfoa", "max"), mxs=("pfos", "max"), n=("pfoa", "size"))
    s = loc.groupby("PWSID").agg(avg_max=("pfoa", "max"), avgs_max=("pfos", "max"), mx=("mx", "max"), mxs=("mxs", "max"), n_samples=("n", "sum"))
    out = pd.DataFrame({"y_avg": ((s.avg_max >= 4.05) | (s.avgs_max >= 4.05)).astype(int),
                        "y_max": ((s.mx >= 4) | (s.mxs >= 4)).astype(int), "n_samples": s.n_samples})
    return out.reset_index()


def load_minnesota():
    d = pd.read_excel(EXT / "minnesota_mdh_20261025/mn_cws_pfas_results.xlsx", "Table", dtype=str)
    d = d[d.Notes.fillna("") != "Purchased Water"].copy()
    d["PWSID"] = "MN" + d.pwsid.str.zfill(7)
    mx = np.maximum(num(d.PFOA_Max), num(d.PFOS_Max)); mn = np.maximum(num(d.PFOA_Min), num(d.PFOS_Min))
    return pd.DataFrame({"PWSID": d.PWSID, "y_max": (mx >= 4).astype(int), "y_min": (mn >= 4).astype(int)}).drop_duplicates("PWSID")


def load_newhampshire():
    d = pd.read_csv(EXT / "newhampshire_nhdes_20261025/nh_pws_pfas_sources.csv", dtype=str, encoding="utf-8-sig")
    d["PWSID"] = "NH" + d.pwsid.str.split("-").str[0].str.zfill(7)
    g = d.groupby("PWSID").agg(y_fed=("fedr_mcl", lambda s: int((s == "Y").any())), y_state=("e_mcl", lambda s: int((s == "Y").any())), n_sources=("pwsid", "size"))
    return g.reset_index()


def boot_ci(fn, y, f, B=2000):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], f[i]))
    return np.percentile(vals, [2.5, 97.5])


def recall_at(y, f, frac):
    k = max(int(round(frac * len(y))), 1); top = np.argsort(-f)[:k]
    return y[top].sum() / max(y.sum(), 1)


if __name__ == "__main__":
    sm = pd.read_parquet(D / "small_frame_anchor.parquet")
    lab = sm[sm.y_mcl.notna()]
    wts = lab.cell.map(sm.groupby("cell").size() / lab.groupby("cell").size())
    k_nat = np.average(lab.y_mcl, weights=wts) / np.average(lab.f, weights=wts)     # national design-weighted recalibration factor
    rect_nat = np.average(lab.y_mcl - lab.f, weights=wts)                           # national design-weighted rectifier
    states = {"MI": (load_michigan(), ["y_avg", "y_max"]), "MN": (load_minnesota(), ["y_max", "y_min"]), "NH": (load_newhampshire(), ["y_fed", "y_state"])}
    rows, prev_rows, conc = [], [], []
    for st, (truth, ends) in states.items():
        frame = sm[sm.PRIMACY_AGENCY_CODE == st]
        m = frame.merge(truth, on="PWSID", how="inner")
        cover = len(m) / len(frame)
        for sub_name, sub in [("unmonitored by UCMR5", m[m.y_mcl.isna()]), ("in UCMR5 small sample", m[m.y_mcl.notna()])]:
            for e in ends:
                y = sub[e].values.astype(int); f = sub.f.values
                if len(y) < 20 or y.sum() < 3 or y.sum() == len(y):
                    rows.append(dict(state=st, subset=sub_name, endpoint=e, n=len(y), positives=int(y.sum()))); continue
                auc = roc_auc_score(y, f); lo, hi = boot_ci(roc_auc_score, y, f)
                rows.append(dict(state=st, subset=sub_name, endpoint=e, n=len(y), positives=int(y.sum()), prevalence=y.mean(),
                                 mean_f=f.mean(), mean_f_recal=np.clip(f * k_nat, 0, 1).mean(), mean_f_rect=np.clip(f + rect_nat, 0, 1).mean(),
                                 auc=auc, auc_lo=lo, auc_hi=hi, pr_auc=average_precision_score(y, f),
                                 recall_top20=recall_at(y, f, 0.2), recall_top30=recall_at(y, f, 0.3), recall_top50=recall_at(y, f, 0.5),
                                 frame_coverage=cover))
        # concordance with the UCMR5 endpoint for systems in both
        both = m[m.y_mcl.notna()]
        for e in ends:
            if len(both):
                conc.append(dict(state=st, endpoint=e, n_both=len(both), ucmr5_prev=both.y_mcl.mean(), state_prev=both[e].mean(),
                                 agree=(both.y_mcl == both[e]).mean(),
                                 sens=(both[e][both.y_mcl == 1] == 1).mean() if (both.y_mcl == 1).any() else np.nan,
                                 spec=(both[e][both.y_mcl == 0] == 0).mean() if (both.y_mcl == 0).any() else np.nan))
        # state-level small-system prevalence: estimators built WITHOUT state data vs the state truth
        e0 = ends[0]; tested = m                                                    # state truth among tested small systems
        truth_prev = tested[e0].mean()
        s_lab = frame[frame.y_mcl.notna()]
        est = {"ML plug-in (census model)": tested.f.mean(),
               "ML recalibrated nationally (x k)": np.clip(tested.f * k_nat, 0, 1).mean(),
               "ML + national rectifier (synthetic PPI)": np.clip(tested.f + rect_nat, 0, 1).mean()}
        des_n = len(s_lab)
        if des_n >= 5:
            # in-state UCMR5 small sample, post-stratified within the state's cells
            g = frame.copy(); g["cell"] = g.cell.where(g.cell.isin(s_lab.cell.unique()), "other")
            th, se, df, n, excl = stratified_burden(g[g.cell != "other"], "systems", False)
            iv = ac.intervals_vec([th], [se], [df], [n])["kg"]
            est["UCMR5 in-state design-only"] = th; ci_design = (iv[0][0], iv[1][0])
            th2, se2, df2, n2, _ = stratified_burden(g[g.cell != "other"], "systems", True)
            iv2 = ac.intervals_vec([th2], [se2], [df2], [n2])["kg"]
            est["UCMR5 in-state PPI++ (state rectifier)"] = th2; ci_ppi = (iv2[0][0], iv2[1][0])
        for k_, v in est.items():
            r = dict(state=st, endpoint=e0, truth_tested_small=truth_prev, n_tested=len(tested), estimator=k_, estimate=v, ratio=v / truth_prev if truth_prev else np.nan,
                     ucmr5_small_n=des_n)
            if k_ == "UCMR5 in-state design-only": r.update(kg_lo=ci_design[0], kg_hi=ci_design[1])
            if k_ == "UCMR5 in-state PPI++ (state rectifier)": r.update(kg_lo=ci_ppi[0], kg_hi=ci_ppi[1])
            prev_rows.append(r)
    S = pd.DataFrame(rows); P = pd.DataFrame(prev_rows); Cc = pd.DataFrame(conc)
    S.to_csv(R / "external_validation_systems.csv", index=False); P.to_csv(R / "external_validation_state_prevalence.csv", index=False)
    Cc.to_csv(R / "external_validation_concordance.csv", index=False)
    lines = ["# External validation against state PFAS data", "", f"National recalibration factor k = {k_nat:.2f}; national rectifier = {rect_nat:+.4f}", "",
             "## System level (census-anchored prior f; no state data used in training)", "", S.round(3).to_markdown(index=False), "",
             "## Agreement of state endpoints with the UCMR5 endpoint (systems in both)", "", Cc.round(3).to_markdown(index=False), "",
             "## State-level small-system prevalence (first endpoint of each state)", "", P.round(4).to_markdown(index=False), ""]
    (R / "external_validation_report.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))
