"""Semi-synthetic projection on the REAL small-system frame (N = 56,591): ANCHOR vs classical risk-stratified Neyman,
the observed-cell reference allocation, population PPS and SRS, for the share of systems and the share of people.

Unsampled small systems have no labels, so pseudo-populations are generated from a truth model fitted to the real
observed cohort with working post-stratification weights:  logit p = a0 + b*logit(f) + size-class and source effects (+ sigma*eps).
This keeps the observed calibration slope and the size-specific under-prediction of the census model. Scenarios:
  fitted     : the weighted fit as estimated (calibration slope 0.67; frame AUC of f vs pseudo-truth ~0.78, conservative)
  auc-match  : slope scaled so the expected frame AUC equals the AUC observed on the real sample (0.847)
  fitted+sd  : fitted plus unit-level risk heterogeneity the census model cannot see (sigma = 0.75; hardest)
Frame prevalence is held at the fitted model's value in every scenario. The planner's inputs (f, u, pc) stay census-trained.
Outputs: results/small_projection.csv, results/small_projection_report.md
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *
import anchor_core as ac

BUDGETS = [400, 800, 1200]
R_DRAWS = 1000
SEEDS = [0, 1, 2, 3, 4]
SCENARIOS = {"fitted": (1.0, 0.0), "auc-match": (None, 0.0), "fitted+sd": (1.0, 0.75)}
TARGET_AUC = 0.847


def truth_model(sm):
    lab = sm[sm.y_mcl.notna()]
    w = lab.cell.map(sm.groupby("cell").size() / lab.groupby("cell").size()).values
    Z = lambda d: np.column_stack([logit(np.clip(d.f.values, 1e-4, 1 - 1e-4))] +
                                  [(d.scls == k).values.astype(float) for k in ["<=100", "101-500", "501-1000"]] + [(d.src == "SW").values.astype(float)])
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(Z(lab), lab.y_mcl.astype(int), sample_weight=w / w.mean())
    return lr.decision_function(Z(sm)), lr


def _recentre(lin, target_mean):
    shift = 0.0
    for _ in range(80):
        p = expit(lin + shift); g = p.mean() - target_mean
        if abs(g) < 1e-7:
            break
        shift -= g / max(np.mean(p * (1 - p)), 1e-6)
    return lin + shift


def expected_auc(f, p):
    r = pd.Series(f).rank().values; n1 = p.sum()
    return (np.sum(p * r) - n1 * (n1 + 1) / 2) / (n1 * (len(f) - n1))


def scenario_linear(sm, eta, lr, scale, sigma, rng):
    """Linear predictor of the pseudo-truth: slope on logit f multiplied by `scale` (None -> match TARGET_AUC)."""
    lf = logit(np.clip(sm.f.values, 1e-4, 1 - 1e-4)); b = lr.coef_[0][0]; target = expit(eta).mean()
    if scale is None:
        lo, hi = 1.0, 4.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if expected_auc(sm.f.values, expit(_recentre(eta + (mid - 1) * b * lf, target))) < TARGET_AUC:
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
    lin = eta + (scale - 1) * b * lf + sigma * rng.standard_normal(len(sm))
    return _recentre(lin, target), scale


def designs(sm, a, base, n, obs_nh):
    N = len(sm); spec = {}
    spec["SRS"] = dict(kind="strat", strata=np.zeros(N, int), Nh=np.array([N], float), nh=np.array([n]))
    cells = sm.cell.astype("category").cat.codes.values; H = cells.max() + 1; Nh = np.bincount(cells, minlength=H).astype(float)
    nh = np.minimum(np.round(obs_nh / obs_nh.sum() * n).astype(int), Nh.astype(int))
    spec["EPA-actual"] = dict(kind="strat", strata=cells, Nh=Nh, nh=nh)
    spec["PPS-pop"] = dict(kind="poisson", pi=ac.cap_probs(sm.P.values, n))
    pc = sm.pc.values; st = ac.rank_strata(a * sm.f.values, a * np.sqrt(pc * (1 - pc)), H=10)
    Nh10 = np.bincount(st, minlength=10).astype(float)
    Sh = np.array([np.sqrt(max(np.mean(a[st == h] ** 2 * pc[st == h]) - np.mean(a[st == h] * pc[st == h]) ** 2, 1e-12)) for h in range(10)])
    spec["Neyman-H10"] = dict(kind="strat", strata=st, Nh=Nh10, nh=ac.neyman_alloc(Nh10, Sh, n))
    spec["ANCHOR"] = dict(kind="poisson", pi=ac.cap_probs(ac.mix_scores(a * sm.u.values, base, 0.5), n))
    for s in spec.values():
        if s["kind"] == "strat":
            s["pi"] = (s["nh"] / s["Nh"])[s["strata"]]
    return spec


def dual_pi(sm, n, rho=0.25):
    u, P, one = sm.u.values, sm.P.values, np.ones(len(sm))
    s = 0.5 * (rho * u / u.mean() + (1 - rho) * P * u / (P * u).mean()) + 0.5 * (rho * one + (1 - rho) * P / P.mean())
    return ac.cap_probs(s, n)


def task(scen, seed):
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); eta, lr = truth_model(sm)
    rng = np.random.default_rng(1000 * seed + list(SCENARIOS).index(scen))
    lin, scale = scenario_linear(sm, eta, lr, *SCENARIOS[scen], rng)
    y = (rng.random(len(sm)) < expit(lin)).astype(float)
    f = sm.f.values; obs_nh = np.bincount(sm.cell.astype("category").cat.codes.values[sm.y_mcl.notna().values])
    out = []; auc = roc_auc_score(y, f)
    for estimand in ["systems", "people"]:
        a = np.ones(len(sm)) if estimand == "systems" else sm.P.values; base = np.ones(len(sm)) if estimand == "systems" else sm.P.values
        A = a.sum(); truth = np.sum(a * y) / A; sumAF = np.sum(a * f)
        for n in BUDGETS:
            spec = designs(sm, a, base, n, obs_nh)
            spec["dual-rho0.25"] = dict(kind="poisson", pi=dual_pi(sm, n))
            for dname, sp in spec.items():
                rec = {"hajek": [], "ppi": []}
                for r in range(R_DRAWS):
                    if sp["kind"] == "strat":
                        m = ac.draw_stratified(sp["strata"], sp["nh"], rng); des = ("strat", sp["strata"][m], sp["Nh"], sp["nh"])
                    else:
                        m = ac.draw_poisson(sp["pi"], rng); des = ("poisson",)
                    ys, w, ps = y[m], 1 / sp["pi"][m], sp["pi"][m]
                    rec["hajek"].append((*ac.est_hajek(ys, a[m], w, ps, A, des), m.sum()))
                    rec["ppi"].append((*ac.est_ppi(ys, f[m], a[m], w, ps, A, sumAF, des), m.sum()))
                for est, rows in rec.items():
                    o = np.array(rows); iv = ac.intervals_vec(o[:, 0], o[:, 1], o[:, 2], o[:, 3])
                    lo, hi = iv["kg"]
                    out.append(dict(scenario=scen, slope_scale=scale, seed=seed, auc_f=auc, estimand=estimand, n=n, design=dname, estimator=est, truth=truth,
                                    bias=o[:, 0].mean() - truth, mse=np.mean((o[:, 0] - truth) ** 2), cov_kg=np.mean((lo <= truth) & (truth <= hi)),
                                    wid_kg=np.mean(hi - lo), cov_wald=np.mean(np.abs(o[:, 0] - truth) <= ac.Z * o[:, 1])))
    print(f"{scen} seed={seed} done; slope x{scale:.2f}; AUC(f vs pseudo-truth)={auc:.3f}", flush=True)
    return out


if __name__ == "__main__":
    import s20_small_projection as mod   # submit the importable module function: loky cannot unpickle scipy ufuncs bound in __main__
    res = Parallel(n_jobs=N_JOBS)(delayed(mod.task)(s, k) for s in SCENARIOS for k in SEEDS)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    key = ["scenario", "seed", "estimand", "n"]
    df = df.join(df[(df.design == "SRS") & (df.estimator == "hajek")].set_index(key).mse.rename("mse_srs"), on=key)
    df = df.join(df[(df.design == "EPA-actual") & (df.estimator == "hajek")].set_index(key).mse.rename("mse_epa"), on=key)
    df["eff_vs_srs"] = df.mse_srs / df.mse; df["eff_vs_epa"] = df.mse_epa / df.mse
    df.to_csv(R / "small_projection.csv", index=False)
    agg = df.groupby(["scenario", "estimand", "n", "design", "estimator"]).agg(eff_vs_srs=("eff_vs_srs", "mean"), eff_vs_epa=("eff_vs_epa", "mean"),
                                                                            cov_kg=("cov_kg", "mean"), cov_wald=("cov_wald", "mean"), auc=("auc_f", "mean")).reset_index()
    agg["label"] = agg.design + " + " + agg.estimator
    lines = ["# Small-frame projection (conditional pseudo-truth fitted to the observed cohort)", ""]
    for sg in SCENARIOS:
        lines.append(f"## scenario {sg} (mean AUC of the census prior vs pseudo-truth {agg[agg.scenario == sg].auc.mean():.3f})")
        for est in ["systems", "people"]:
            t = agg[(agg.scenario == sg) & (agg.estimand == est)].pivot_table(index="label", columns="n", values="eff_vs_epa").round(2)
            lines += [f"### {est}: efficiency vs EPA-actual allocation + Hajek (MSE ratio)", "", t.to_markdown(), ""]
    k = ["scenario", "estimand", "n"]
    an = agg[agg.label == "ANCHOR + ppi"].set_index(k).eff_vs_epa
    cmp_ = pd.DataFrame({lab: agg[agg.label == lab].set_index(k).eff_vs_epa / an for lab in ["Neyman-H10 + hajek", "Neyman-H10 + ppi", "dual-rho0.25 + ppi", "EPA-actual + ppi"]}).round(2)
    lines += ["## Competitor / ANCHOR efficiency ratio", "", cmp_.to_markdown(), ""]
    e800 = agg[agg.n == 800].assign(n_equal_to_EPA800=lambda d: (800 / d.eff_vs_epa).round(0)).pivot_table(index=["label"], columns=["scenario", "estimand"], values="n_equal_to_EPA800")
    lines += ["## Sample size needed to match the precision of the EPA-actual allocation at n = 800", "", e800.to_markdown(), ""]
    cv = agg.pivot_table(index="label", columns=["scenario", "estimand"], values="cov_kg", aggfunc="min").round(3)
    lines += ["## KG coverage (minimum over budgets)", "", cv.to_markdown(), ""]
    (R / "small_projection_report.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))
