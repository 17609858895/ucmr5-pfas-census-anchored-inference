"""Follow-up to the external validation: how much of the national discrimination is within-state, and does a
national multiplicative recalibration transport to state-level small-system prevalence (with a bootstrap interval)?

Ratio (multiplicative) state estimator: theta_state = k * mean_state(f), with k = design-weighted mean(y) / mean(f) on the
UCMR5 small-system probability sample, bootstrapped within reconstructed cells. Logit-shift estimator:
theta_state = mean_state expit(logit f + delta), delta chosen so the design-weighted sample mean of expit(logit f + delta)
equals the design-weighted observed prevalence (bounded in [0, 1]; equals the ratio estimator for small f).
Both compared with the additive (difference-type, PPI-style) transfer mean_state(f) + national rectifier. Truth = state data (s21), never used here.
Run with `python -P`. Output: results/external_validation_followup.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from common import *


def within_group_auc(df, g):
    conc = tot = 0.0
    for _, d in df.groupby(g):
        P, Nn = d.f.values[d.y_mcl.values == 1], d.f.values[d.y_mcl.values == 0]
        if len(P) and len(Nn):
            conc += (P[:, None] > Nn[None, :]).sum() + 0.5 * (P[:, None] == Nn[None, :]).sum(); tot += len(P) * len(Nn)
    return conc / tot


if __name__ == "__main__":
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); lab = sm[sm.y_mcl.notna()].copy()
    rows = [dict(item="UCMR5 small sample: pooled AUC", value=roc_auc_score(lab.y_mcl, lab.f)),
            dict(item="UCMR5 small sample: within-state AUC", value=within_group_auc(lab, "PRIMACY_AGENCY_CODE")),
            dict(item="UCMR5 small sample: within-EPA-region AUC", value=within_group_auc(lab, "EPA_REGION"))]
    w = lab.cell.map(sm.groupby("cell").size() / lab.groupby("cell").size()).values
    k = np.average(lab.y_mcl, weights=w) / np.average(lab.f, weights=w); rect = np.average(lab.y_mcl - lab.f, weights=w)
    rng = np.random.default_rng(1); cells = lab.cell.values; ks, rs = [], []
    for b in range(4000):
        idx = np.concatenate([rng.choice(np.flatnonzero(cells == c), (cells == c).sum(), replace=True) for c in np.unique(cells)])
        yy, ff, ww = lab.y_mcl.values[idx], lab.f.values[idx], w[idx]
        ks.append(np.average(yy, weights=ww) / np.average(ff, weights=ww)); rs.append(np.average(yy - ff, weights=ww))
    klo, khi = np.percentile(ks, [2.5, 97.5]); rlo, rhi = np.percentile(rs, [2.5, 97.5])
    lf_lab = logit(np.clip(lab.f.values, 1e-5, 1 - 1e-5))
    shift = lambda yy, ll, ww: brentq(lambda d: np.average(expit(ll + d), weights=ww) - np.average(yy, weights=ww), -5, 5)
    d0 = shift(lab.y_mcl.values, lf_lab, w); ds = []
    for b in range(2000):
        idx = np.concatenate([rng.choice(np.flatnonzero(cells == c), (cells == c).sum(), replace=True) for c in np.unique(cells)])
        ds.append(shift(lab.y_mcl.values[idx], lf_lab[idx], w[idx]))
    dlo, dhi = np.percentile(ds, [2.5, 97.5])
    rows += [dict(item="national recalibration factor k", value=k, lo=klo, hi=khi), dict(item="national additive rectifier", value=rect, lo=rlo, hi=rhi),
             dict(item="national logit shift delta", value=d0, lo=dlo, hi=dhi)]
    prev = pd.read_csv(R / "external_validation_state_prevalence.csv")
    for st in ["MI", "MN"]:
        truth = prev[prev.state == st].truth_tested_small.iloc[0]; fs = sm[sm.PRIMACY_AGENCY_CODE == st].f
        rows.append(dict(item=f"{st}: state truth (tested small systems)", value=truth))
        rows.append(dict(item=f"{st}: ratio estimator k*mean(f)", value=k * fs.mean(), lo=klo * fs.mean(), hi=khi * fs.mean(), ratio_to_truth=k * fs.mean() / truth))
        rows.append(dict(item=f"{st}: additive estimator mean(f)+rectifier", value=fs.mean() + rect, lo=fs.mean() + rlo, hi=fs.mean() + rhi, ratio_to_truth=(fs.mean() + rect) / truth))
        lfs = logit(np.clip(fs.values, 1e-5, 1 - 1e-5)); est = expit(lfs + d0).mean()
        rows.append(dict(item=f"{st}: logit-shift estimator", value=est, lo=expit(lfs + dlo).mean(), hi=expit(lfs + dhi).mean(), ratio_to_truth=est / truth))
        rows.append(dict(item=f"{st}: plug-in mean(f)", value=fs.mean(), ratio_to_truth=fs.mean() / truth))
    out = pd.DataFrame(rows); out.to_csv(R / "external_validation_followup.csv", index=False)
    print(out.round(4).to_string(index=False))
