"""Two-level budget: which systems to sample x how many samples each, under a fixed number of analytical samples.

Within-system rules (s23): full schedule; R1 (stop a location after one non-detect); R12 (R1 at groundwater locations,
two consecutive non-detects at surface-water locations). Truncation can only miss exceedances (y_tilde <= y).
Two-phase correction: a random fraction q of sampled systems completes the full schedule regardless of the rule, and
  theta_hat = [lam * sum_U a f + sum_s w a (y_tilde - lam f) + sum_s2 (w/q) a (y - y_tilde)] / sum_U a
(lam = 0 gives the design-only version). Variance: Poisson approximation
  V = [sum_s w(w-1) e^2 + sum_s2 (w/q){(w-1)(2 e d + d^2) + w(1/q - 1) d^2}] / A^2,  e = a(y_tilde - lam f), d = a(y - y_tilde).
Part A: census truth (medium, large) with real schedules. Part B: small-frame projection with donor schedules drawn from the
970 UCMR5 small systems with the same (exceedance, source type). Outputs: results/two_level_census.csv, two_level_small.csv
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from common import *
import anchor_core as ac

def budget_probs(score, cost, budget):
    """Calibrate capped inclusion probabilities to expected analytical cost."""
    score=np.asarray(score,float); cost=np.asarray(cost,float)
    if budget >= cost.sum(): return np.ones(len(score))
    lo,hi=0.,budget/max(np.sum(score*cost),1e-12)
    while np.sum(np.minimum(1,hi*score)*cost)<budget: hi*=2
    for _ in range(70):
        mid=(lo+hi)/2
        if np.sum(np.minimum(1,mid*score)*cost)<budget: lo=mid
        else: hi=mid
    pi=np.minimum(1,(lo+hi)/2*score)
    assert np.isclose(np.sum(pi*cost),budget,rtol=1e-10)
    return pi


RULE_COLS = ["cost_full", "cost_R1", "cost_R12", "yt_R1", "yt_R12"]


def estimate(y, yt, f, a, pi, I, J, q, A, sumAF, ppi):
    w = 1 / pi[I]; ys, yts, fs, as_ = y[I], yt[I], f[I], a[I]; j = J[I]
    if ppi:
        yb = np.sum(w * as_ * yts) / np.sum(w * as_); fb = np.sum(w * as_ * fs) / np.sum(w * as_)
        lam = np.clip(np.sum(w * as_ ** 2 * (yts - yb) * (fs - fb)) / max(np.sum(w * as_ ** 2 * (fs - fb) ** 2), 1e-12), 0, 2)
    else:
        lam = 0.0
    e = as_ * (yts - lam * fs); dlt = as_ * (ys - yts) * j
    corr = np.sum(w[j] / q * dlt[j]) if q > 0 else 0.0
    th = (lam * sumAF + np.sum(w * e) + corr) / A
    v = np.sum(w * (w - 1) * e ** 2)
    if q > 0:
        wj, ej, dj = w[j], e[j], dlt[j]
        v += np.sum(wj / q * ((wj - 1) * (2 * ej * dj + dj ** 2) + wj * (1 / q - 1) * dj ** 2))
    return th, np.sqrt(max(v, 0)) / A, max(I.sum() - 2, 1), I.sum()


def run_strategy(rng, y, f, u, a, costs, design, rule, q, ppi, B, R, base=None, strata=None, obs_nh=None, P=None):
    N = len(y); c_rule = costs[f"cost_{rule}"] if rule != "full" else costs["cost_full"]
    yt = costs[f"yt_{rule}"] if rule != "full" else y
    c_unit = q * costs["cost_full"] + (1 - q) * c_rule if rule != "full" else costs["cost_full"]
    if design == "ANCHOR":
        s = ac.mix_scores(a * u, base, .5); pi = budget_probs(s, c_unit, B)
    elif design == "dual":
        assert P is not None
        s = .5 * (.25 * u / u.mean() + .75 * P * u / (P * u).mean()) + .5 * (.25 + .75 * P / P.mean())
        pi = budget_probs(s, c_unit, B)
    elif design == "SRS":
        pi = budget_probs(np.ones(N), c_unit, B)
    elif design == "EPA":
        nh_share = obs_nh / obs_nh.sum(); Nh = np.bincount(strata, minlength=len(obs_nh)).astype(float)
        pi = budget_probs((nh_share / Nh)[strata], c_unit, B)
    A = a.sum(); sumAF = np.sum(a * f); truth = np.sum(a * y) / A
    rows = []
    for r in range(R):
        I = rng.random(N) < pi; J = rng.random(N) < q
        th, se, df, nn = estimate(y, yt, f, a, pi, I, J, q, A, sumAF, ppi)
        cost = np.sum(np.where(J[I], costs["cost_full"][I], c_rule[I])) if rule != "full" else np.sum(costs["cost_full"][I])
        rows.append((th, se, df, nn, cost))
    o = np.array(rows); lo, hi = ac.intervals_vec(o[:, 0], o[:, 1], o[:, 2], o[:, 3])["kg"]
    return dict(truth=truth, bias=o[:, 0].mean() - truth, mse=np.mean((o[:, 0] - truth) ** 2), cov_kg=np.mean((lo <= truth) & (truth <= hi)),
                n_mean=o[:, 3].mean(), cost_mean=o[:, 4].mean(), expected_cost=float(np.sum(pi*c_unit)))


STRATS_A = [("Equal-probability Poisson · full · HT", "SRS", "full", 0, False), ("Equal-probability Poisson · R12 · HT", "SRS", "R12", 0, False),
            ("ANCHOR · full", "ANCHOR", "full", 0, True), ("ANCHOR · R12", "ANCHOR", "R12", 0, True),
            ("ANCHOR · R12 + two-phase q=0.2", "ANCHOR", "R12", .2, True), ("ANCHOR · R1 + two-phase q=0.1", "ANCHOR", "R1", .1, True),
            ("ANCHOR · R1 + two-phase q=0.2", "ANCHOR", "R1", .2, True), ("ANCHOR · R1 + two-phase q=0.3", "ANCHOR", "R1", .3, True),
            ("ANCHOR · R1", "ANCHOR", "R1", 0, True)]


def census_task(popname, seed, R=1000):
    F = pd.read_parquet(D / "frame_features.parquet")
    pop = F[(F.size_cat == popname) & F.y_mcl.notna() & F.lat.notna() & F["pop"].notna()].reset_index(drop=True)
    d = pickle.loads((D / "sim_pops" / f"{popname}_geo+nb_{seed}.pkl").read_bytes())
    W = pd.read_parquet(R_ / "within_system_rules.parquet").set_index("PWSID").reindex(pop.PWSID)
    assert W.y.notna().all() and np.array_equal(W.y.values.astype(float), d["y"])
    costs = {c: W[c].values.astype(float) for c in RULE_COLS}
    rng = np.random.default_rng(700 + seed); out = []
    for estimand in ["systems", "people"]:
        a = np.ones(len(d["y"])) if estimand == "systems" else d["P"]
        for n0 in [150, 300, 600]:
            B = n0 * costs["cost_full"].mean()
            for name, design, rule, q, ppi in STRATS_A:
                r = run_strategy(rng, d["y"], d["f"], d["u"], a, costs, design, rule, q, ppi, B, R, base=a)
                out.append(dict(population=popname, seed=seed, estimand=estimand, budget=B, n0=n0, strategy=name, **r))
    return out


STRATS_B = [("EPA allocation · full", "EPA", "full", 0, False), ("EPA allocation · R12 + two-phase", "EPA", "R12", .2, False),
            ("ANCHOR · full", "ANCHOR", "full", 0, True), ("ANCHOR · R12 + two-phase", "ANCHOR", "R12", .2, True),
            ("ANCHOR · R1 + two-phase", "ANCHOR", "R1", .2, True), ("Dual · R12 + two-phase", "dual", "R12", .2, True)]


def small_task(scen, seed, R=600):
    import s20_small_projection as sp
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); eta, lr = sp.truth_model(sm)
    rng = np.random.default_rng(900 + 10 * seed + list(sp.SCENARIOS).index(scen))
    lin, _ = sp.scenario_linear(sm, eta, lr, *sp.SCENARIOS[scen], rng)
    y = (rng.random(len(sm)) < expit(lin)).astype(float)
    W = pd.read_parquet(R_ / "within_system_rules.parquet"); W = W[W.size_cat == "small"].merge(sm[["PWSID", "src"]], on="PWSID")
    costs = {c: np.zeros(len(sm)) for c in RULE_COLS}
    for (yy, src), g in W.groupby(["y", "src"]):
        idx = np.flatnonzero((y == yy) & (sm.src.values == src))
        pick = g.iloc[rng.integers(0, len(g), len(idx))]
        for c in RULE_COLS:
            costs[c][idx] = pick[c].values
    cells = sm.cell.astype("category").cat.codes.values
    obs_nh = np.bincount(cells[sm.y_mcl.notna().values], minlength=cells.max() + 1).astype(float)
    out = []
    for estimand in ["systems", "people"]:
        a = np.ones(len(sm)) if estimand == "systems" else sm.P.values
        for B in [1500, 3000]:
            for name, design, rule, q, ppi in STRATS_B:
                draw_rng = np.random.default_rng(62000 + 1000*seed + 100*list(sp.SCENARIOS).index(scen) + int(B)) if design == 'dual' else rng
                r = run_strategy(draw_rng, y, sm.f.values, sm.u.values, a, costs, design, rule, q, ppi, B, R, base=a, strata=cells, obs_nh=obs_nh, P=sm.P.values)
                out.append(dict(scenario=scen, seed=seed, estimand=estimand, budget=B, strategy=name, **r))
        # reference: current practice = EPA allocation, full schedule, 800 systems
        base = run_strategy(rng, y, sm.f.values, sm.u.values, a, costs, "EPA", "full", 0, False,
                            800 * np.sum(obs_nh / obs_nh.sum() * np.array([costs["cost_full"][cells == h].mean() for h in range(len(obs_nh))])),
                            R, base=a, strata=cells, obs_nh=obs_nh)
        out.append(dict(scenario=scen, seed=seed, estimand=estimand, budget=base["expected_cost"], strategy="REFERENCE: EPA 800 full", **base))
    return out


R_ = R
if __name__ == "__main__":
    import s24_two_level as mod
    resA = Parallel(n_jobs=N_JOBS)(delayed(mod.census_task)(p, s) for p in ["medium", "large"] for s in range(5))
    A_ = pd.DataFrame([r for c in resA for r in c]); A_.to_csv(R / "two_level_census.csv", index=False); print("part A saved", flush=True)
    resB = Parallel(n_jobs=N_JOBS)(delayed(mod.small_task)(sc, s) for sc in ["fitted", "auc-match"] for s in range(5))
    B_ = pd.DataFrame([r for c in resB for r in c]); B_.to_csv(R / "two_level_small.csv", index=False); print("part B saved", flush=True)
