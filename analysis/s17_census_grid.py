"""Census-truth grid: ANCHOR vs classical risk-stratified Neyman, EPA-like population PPS, SRS and model-assisted GREG.

Truth populations: MEDIUM census (prior trained on LARGE) and LARGE census (prior trained on MEDIUM).
Before sampling only the training census labels are used: the prior model f, its cross-fitted error model u,
the neighbour-label features, and (for the classical Neyman design) census-calibrated stratum variances.
Outputs: results/grid_cheap.parquet (all designs x closed-form estimators) and results/grid_gregml.parquet.
Usage:  python s17_census_grid.py [--quick]
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import pickle
from common import *
from model_features import nb_feats, BASE, Q
import anchor_core as ac

QUICK = "--quick" in sys.argv
SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
BUDGETS = [150, 300] if QUICK else [100, 150, 300, 600, 1000]
R_CHEAP = 200 if QUICK else 1000
R_ML = 30 if QUICK else 100   # GREG-ML refits 5 boosters per draw; 100 x 5 seeds = 500 draws per configuration
AUX = ["lpop", "sw", "lat", "lon", "cws", "purch", "l_sources", "q_kmcl", "q_mcl25", "l_dmin"]
GB = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, verbose=-1, n_jobs=2)


def build_population(popname, trainname, prior, seed):
    F = pd.read_parquet(D / "frame_features.parquet")
    tr = F[(F.size_cat == trainname) & F.y_mcl.notna() & F.lat.notna()].copy()
    pop = F[(F.size_cat == popname) & F.y_mcl.notna() & F.lat.notna() & F["pop"].notna()].copy().reset_index(drop=True)
    tr = tr.join(nb_feats(tr, tr)); pop = pop.join(nb_feats(pop, tr))
    X = BASE + (Q if prior == "geo+nb" else [])
    ytr = tr.y_mcl.values.astype(int)
    f = lgb.LGBMClassifier(**GB, random_state=seed).fit(tr[X], ytr).predict_proba(pop[X])[:, 1]
    oof = np.zeros(len(tr))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(tr):
        oof[b] = lgb.LGBMClassifier(**GB, random_state=seed).fit(tr[X].iloc[a], ytr[a]).predict_proba(tr[X].iloc[b])[:, 1]
    u = np.clip(lgb.LGBMRegressor(**GB, random_state=seed).fit(tr[X], np.abs(ytr - oof)).predict(pop[X]), 1e-3, None)
    pc = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=0.999).fit(oof, ytr).predict(f)   # census-calibrated risk
    pop["l_sources"] = np.log1p(pop.n_sources); pop["l_dmin"] = np.log1p(pop.q_dmin.fillna(50))
    pop["q_mcl25"] = pop.q_mcl25.fillna(pop.q_kmcl); pop["q_kmcl"] = pop.q_kmcl.fillna(pop.q_kmcl.median())
    Xa = pop[AUX].astype(float); Xa = Xa.fillna(Xa.median()).values
    Xm = pop[BASE + Q].astype(float).values
    return dict(pop=popname, prior=prior, seed=seed, y=pop.y_mcl.values.astype(float), P=pop["pop"].values.astype(float),
                f=f, u=u, pc=pc, sd=np.sqrt(np.clip(f, 1e-3, 1) * (1 - np.clip(f, 1e-3, 1))), Xa=Xa, Xm=Xm,
                auc=float(roc_auc_score(pop.y_mcl.values, f)))


CACHE = D / "sim_pops"


def get_population(popname, trainname, prior, seed):
    """Build once and cache, so that every task for the same population/prior/seed shares one prior model."""
    CACHE.mkdir(exist_ok=True); fp = CACHE / f"{popname}_{prior}_{seed}.pkl"
    if fp.exists():
        return pickle.loads(fp.read_bytes())
    d = build_population(popname, trainname, prior, seed); fp.write_bytes(pickle.dumps(d))
    return d


def weights_for(d, estimand):
    return np.ones(len(d["y"])) if estimand == "systems" else d["P"]


def neyman_spec(d, a, n, H):
    pc = d["pc"]
    # order by census-calibrated risk score (ties broken by the continuous prior f), equal-count strata
    strata = ac.rank_strata(a * d["f"], a * np.sqrt(pc * (1 - pc)), H=H)
    Nh = np.bincount(strata, minlength=H).astype(float)
    Sh = np.array([np.sqrt(max(np.mean(a[strata == h] ** 2 * pc[strata == h]) - np.mean(a[strata == h] * pc[strata == h]) ** 2, 1e-12)) for h in range(H)])
    nh = ac.neyman_alloc(Nh, Sh, n)
    return dict(kind="strat", strata=strata, Nh=Nh, nh=nh, pi=(nh / Nh)[strata])


def designs_for(d, estimand, n, extra=False):
    a = weights_for(d, estimand); N = len(a); one = np.ones(N)
    base = one if estimand == "systems" else a
    specs = {"SRS": dict(kind="strat", strata=np.zeros(N, int), Nh=np.array([N], float), nh=np.array([n]), pi=np.full(N, n / N)),
             "PPS-pop": dict(kind="poisson", pi=ac.cap_probs(d["P"], n)),
             "Neyman-H5": neyman_spec(d, a, n, 5), "Neyman-H10": neyman_spec(d, a, n, 10),
             "ANCHOR": dict(kind="poisson", pi=ac.cap_probs(ac.mix_scores(a * d["u"], base, 0.5), n)),
             "ANCHOR-sys": dict(kind="sys", pi=ac.cap_probs(ac.mix_scores(a * d["u"], base, 0.5), n)),
             "ANCHOR-sd": dict(kind="poisson", pi=ac.cap_probs(ac.mix_scores(a * d["sd"], base, 0.5), n))}
    if extra:
        for tau in [0.25, 0.75]:
            specs[f"ANCHOR-tau{tau}"] = dict(kind="poisson", pi=ac.cap_probs(ac.mix_scores(a * d["u"], base, tau), n))
    return specs


def draw(spec, rng):
    if spec["kind"] == "strat":
        m = ac.draw_stratified(spec["strata"], spec["nh"], rng); des = ("strat", spec["strata"][m], spec["Nh"], spec["nh"])
    elif spec["kind"] == "sys":
        m = ac.draw_systematic(spec["pi"], rng); des = ("poisson",)
    else:
        m = ac.draw_poisson(spec["pi"], rng); des = ("poisson",)
    return m, des


def run_estimators(d, a, m, des, pi, names, seed=0):
    y = d["y"][m]; w = 1 / pi[m]; A = a.sum(); out = {}
    for nm in names:
        if nm == "hajek":
            out[nm] = ac.est_hajek(y, a[m], w, pi[m], A, des)
        elif nm == "ppi":
            out[nm] = ac.est_ppi(y, d["f"][m], a[m], w, pi[m], A, np.sum(a * d["f"]), des)
        elif nm == "ppi_lam1":
            out[nm] = ac.est_ppi(y, d["f"][m], a[m], w, pi[m], A, np.sum(a * d["f"]), des, lam=1.0)
        elif nm == "greg_lin":
            out[nm] = ac.est_greg(y, a[m], w, pi[m], A, des, d["Xa"][m], d["Xa"], a, "lin")
        elif nm == "greg_ml":
            out[nm] = ac.est_greg(y, a[m], w, pi[m], A, des, d["Xm"][m], d["Xm"], a, "ml", seed=seed)
    return out


def summarise(rows, truth):
    th = np.array([r[0] for r in rows]); se = np.array([r[1] for r in rows]); df = np.array([r[2] for r in rows]); nn = np.array([r[3] for r in rows])
    res = dict(truth=truth, bias=th.mean() - truth, rmse=np.sqrt(np.mean((th - truth) ** 2)), emp_sd=th.std(),
               mean_se=np.sqrt(np.mean(se ** 2)), n_real=nn.mean(), draws=len(th))
    for iv, (lo, hi) in ac.intervals_vec(th, se, df, nn).items():
        res[f"cov_{iv}"] = np.mean((lo <= truth) & (truth <= hi)); res[f"wid_{iv}"] = np.mean(hi - lo)
    return res


EST_BY_DESIGN = {"SRS": ["hajek", "ppi", "greg_lin"], "PPS-pop": ["hajek", "ppi"], "Neyman-H5": ["hajek", "ppi", "greg_lin"],
                 "Neyman-H10": ["hajek", "ppi"], "ANCHOR": ["hajek", "ppi", "greg_lin", "ppi_lam1"], "ANCHOR-sys": ["ppi"],
                 "ANCHOR-sd": ["ppi"], "ANCHOR-tau0.25": ["ppi"], "ANCHOR-tau0.75": ["ppi"]}


def cheap_task(popname, trainname, prior, seed):
    t0 = time.time(); d = get_population(popname, trainname, prior, seed); rng = np.random.default_rng(1000 + seed); out = []
    for estimand in ["systems", "people"]:
        a = weights_for(d, estimand); truth = np.sum(a * d["y"]) / a.sum()
        for n in BUDGETS:
            for dname, spec in designs_for(d, estimand, n, extra=(n == 300)).items():
                rec = {e: [] for e in EST_BY_DESIGN[dname]}
                for r in range(R_CHEAP):
                    m, des = draw(spec, rng)
                    for e, v in run_estimators(d, a, m, des, spec["pi"], EST_BY_DESIGN[dname]).items():
                        rec[e].append((*v, m.sum()))
                for e, rows in rec.items():
                    out.append(dict(population=popname, prior=prior, seed=seed, estimand=estimand, n=n, design=dname, estimator=e,
                                    auc=d["auc"], **summarise(rows, truth)))
    # Pareto: one dual-purpose design, both estimands evaluated on the same draws
    one = np.ones(len(d["y"])); P = d["P"]
    for n in [300, 600]:
        for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = 0.5 * (rho * d["u"] / d["u"].mean() + (1 - rho) * P * d["u"] / (P * d["u"]).mean()) + 0.5 * (rho * one + (1 - rho) * P / P.mean())
            pi = ac.cap_probs(s, n); rec = {"systems": [], "people": []}
            for r in range(R_CHEAP):
                m = ac.draw_poisson(pi, rng)
                for estimand in rec:
                    a = weights_for(d, estimand)
                    rec[estimand].append((*ac.est_ppi(d["y"][m], d["f"][m], a[m], 1 / pi[m], pi[m], a.sum(), np.sum(a * d["f"]), ("poisson",)), m.sum()))
            for estimand, rows in rec.items():
                a = weights_for(d, estimand)
                out.append(dict(population=popname, prior=prior, seed=seed, estimand=estimand, n=n, design=f"dual-rho{rho}", estimator="ppi",
                                auc=d["auc"], **summarise(rows, np.sum(a * d["y"]) / a.sum())))
    print(f"cheap {popname}/{prior}/seed{seed} done in {time.time()-t0:.0f}s", flush=True)
    return out


def ml_task(popname, trainname, seed, estimand, n, dname):
    d = get_population(popname, trainname, "geo+nb", seed); rng = np.random.default_rng(5000 + seed * 97 + n)
    a = weights_for(d, estimand); spec = designs_for(d, estimand, n)[dname]; rows = []
    for r in range(R_ML):
        m, des = draw(spec, rng)
        rows.append((*run_estimators(d, a, m, des, spec["pi"], ["greg_ml"], seed=seed * 1000 + r)["greg_ml"], m.sum()))
    return dict(population=popname, prior="geo+nb", seed=seed, estimand=estimand, n=n, design=dname, estimator="greg_ml",
                auc=d["auc"], **summarise(rows, np.sum(a * d["y"]) / a.sum()))


if __name__ == "__main__":
    tag = "_quick" if QUICK else ""
    pairs = [("medium", "large"), ("large", "medium")]
    # build every cached population once, serially, so parallel tasks never race on the same cache file
    for p, t in pairs:
        for pr in ["geo", "geo+nb"]:
            for s in SEEDS:
                get_population(p, t, pr, s)
    print("populations ready", flush=True)
    if "--ml-only" not in sys.argv:
        jobs = [delayed(cheap_task)(p, t, pr, s) for p, t in pairs for pr in ["geo", "geo+nb"] for s in SEEDS]
        res = Parallel(n_jobs=min(len(jobs), N_JOBS), verbose=0)(jobs)
        pd.DataFrame([r for chunk in res for r in chunk]).to_parquet(R / f"grid_cheap{tag}.parquet", index=False)
        print("cheap grid saved", flush=True)
    if "--cheap-only" not in sys.argv:
        jobs = [delayed(ml_task)(p, t, s, e, n, dn) for p, t in pairs for s in SEEDS for e in ["systems", "people"] for n in BUDGETS
                for dn in ["SRS", "Neyman-H5", "ANCHOR"]]
        res = Parallel(n_jobs=N_JOBS, verbose=0)(jobs)   # 7 workers: each loads pandas+LightGBM (~350 MB); 16 GB RAM machine
        pd.DataFrame(res).to_parquet(R / f"grid_gregml{tag}.parquet", index=False)
        print("GREG-ML grid saved", flush=True)
