"""Figure data snapshots for Fig. 1-7 (read by the f0x_*.py figure scripts; Fig. 4 reads the grid summary directly).

Reads retained outputs (data_derived/, results/) and recomputes only light-weight summaries: ROC curves of two
ablation models (census-trained, one seed), census out-of-fold predictions for the size curve, and state-level
ratio estimates k * mean_state(f). Run with `python -P` (it reads the external state downloads via s21).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import expit, logit
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
from common import *
from s19_small_frame_burden import build_table, BASE, GB
import s21_external_states as ext
import anchor_core as ac

OUT = HERE / "figure_data"; OUT.mkdir(exist_ok=True)
CLASSES = [(-1, 100, "≤100"), (100, 500, "101–500"), (500, 1000, "501–1,000"), (1000, 3300, "1,001–3,300"),
           (3300, 10000, "3,301–10,000"), (10000, 1e10, ">10,000")]


def size_class(pop):
    lab = pd.Series("", index=pop.index)
    for lo, hi, name in CLASSES:
        lab[(pop > lo) & (pop <= hi)] = name
    return lab


def fig1(F):
    cat = np.where(~F.in_ucmr5, "Unmonitored", np.where(F.size_cat == "small", "Sampled small (<3,300)", "Census (≥3,300)"))
    F[["PWSID", "lat", "lon"]].assign(category=cat).to_csv(OUT / "fig1_map_points.csv", index=False)
    F = F.assign(cls=size_class(F["pop"].fillna(0)), mon=F.in_ucmr5.astype(int), P=F["pop"].fillna(0))
    cov = F.groupby("cls").agg(systems=("PWSID", "size"), monitored=("mon", "sum"), people=("P", "sum"),
                               people_monitored=("P", lambda s: s[F.loc[s.index, "mon"] == 1].sum()))
    cov = cov.reindex([c[2] for c in CLASSES]).reset_index()
    cov["share_systems"] = cov.monitored / cov.systems; cov["share_people"] = cov.people_monitored / cov.people
    cov.to_csv(OUT / "fig1_coverage_by_class.csv", index=False)
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); sm["cls"] = size_class(sm["P"])
    lab = sm[sm.y_mcl.notna()]
    comp = pd.DataFrame({"Frame: systems": sm.groupby("cls").size() / len(sm),
                         "Frame: people": sm.groupby("cls").P.sum() / sm.P.sum(),
                         "UCMR5 sample: systems": lab.groupby("cls").size() / len(lab)}).reindex([c[2] for c in CLASSES[:4]])
    comp.to_csv(OUT / "fig1_small_composition.csv")


def fig2(T, X):
    sm = pd.read_parquet(D / "small_frame_anchor.parquet")
    cen = T[T.size_cat.isin(["medium", "large"]) & T.y_mcl.notna()].reset_index(drop=True)
    small = T[T.size_cat == "small"].reset_index(drop=True).merge(sm[["PWSID", "f"]], on="PWSID")
    y = cen.y_mcl.values.astype(int)
    sets = {"SDWIS attributes": [c for c in BASE if c not in ("lat", "lon", "state", "region")], "+ geography": BASE}
    lab = small[small.y_mcl.notna()]; roc = []
    for name, cols in sets.items():
        f = lgb.LGBMClassifier(**GB, random_state=0).fit(cen[cols], y).predict_proba(lab[cols])[:, 1]
        fpr, tpr, _ = roc_curve(lab.y_mcl, f); roc.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr, "auc": roc_auc_score(lab.y_mcl, f)}))
    fpr, tpr, _ = roc_curve(lab.y_mcl, lab.f)
    roc.append(pd.DataFrame({"model": "+ census neighbour labels", "fpr": fpr, "tpr": tpr, "auc": roc_auc_score(lab.y_mcl, lab.f)}))
    pd.concat(roc).to_csv(OUT / "fig2_roc.csv", index=False)
    # census out-of-fold prediction vs observed along population, plus small-frame prediction
    oof = np.zeros(len(cen))
    for a, b in KFold(5, shuffle=True, random_state=0).split(cen):
        oof[b] = lgb.LGBMClassifier(**GB, random_state=0).fit(cen[X].iloc[a], y[a]).predict_proba(cen[X].iloc[b])[:, 1]
    cen = cen.assign(f=oof, grp="census"); small = small.assign(grp="small")
    edges = np.r_[np.log10(25), np.log10([100, 300, 1000, 3300, 6000, 10000, 30000, 100000, 1e6]), 7.5]
    rows = []
    for g, d in [("census", cen), ("small", small)]:
        b = np.digitize(np.log10(d["pop"].clip(lower=25)), edges[1:-1])
        for k, dd in d.groupby(b):
            r = dict(group=g, lpop=np.log10(dd["pop"].clip(lower=25)).median(), n=len(dd), pred=dd.f.mean())
            if g == "census":
                p = dd.y_mcl.mean(); z = 1.96; n = len(dd)
                c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                r.update(obs=p, lo=c - h, hi=c + h)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "fig2_size_curve.csv", index=False)
    bur = pd.read_csv(R / "small_burden_kg.csv")
    w = lab.assign(cls=size_class(lab["pop"].fillna(sm.P.median())))
    cls = []
    for name in [c[2] for c in CLASSES[:4]]:
        key = "size " + {"≤100": "<=100", "101–500": "101-500", "501–1,000": "501-1000", "1,001–3,300": "1001-3300"}[name]
        r = bur[(bur.group == key) & (bur.estimand == "systems")].set_index("estimator")
        cls.append(dict(cls=name, obs=r.loc["design-only (post-stratified Hajek)", "theta"], lo=r.loc["design-only (post-stratified Hajek)", "kg_lo"],
                        hi=r.loc["design-only (post-stratified Hajek)", "kg_hi"], plugin=r.loc["ML plug-in (census model, no rectification)", "theta"],
                        lpop=np.log10(w[w.cls == name]["pop"].clip(lower=25)).median()))
    pd.DataFrame(cls).to_csv(OUT / "fig2_by_class.csv", index=False)


def fig5():
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); lab = sm[sm.y_mcl.notna()]
    w = lab.cell.map(sm.groupby("cell").size() / lab.groupby("cell").size()).values
    k = np.average(lab.y_mcl, weights=w) / np.average(lab.f, weights=w)
    rng = np.random.default_rng(1); cells = lab.cell.values; ks = []
    for b in range(4000):
        idx = np.concatenate([rng.choice(np.flatnonzero(cells == c), (cells == c).sum(), replace=True) for c in np.unique(cells)])
        ks.append(np.average(lab.y_mcl.values[idx], weights=w[idx]) / np.average(lab.f.values[idx], weights=w[idx]))
    klo, khi = np.percentile(ks, [2.5, 97.5])
    fu = pd.read_csv(R / "external_validation_followup.csv").set_index("item")
    d0, dlo, dhi = fu.loc["national logit shift delta", ["value", "lo", "hi"]]
    lf = logit(np.clip(sm.f.values, 1e-5, 1 - 1e-5))
    for nm, d in [("p", d0), ("p_lo", dlo), ("p_hi", dhi)]:
        sm[nm] = expit(lf + d)                      # logit-shift recalibration (bounded; validated in MI and MN)
    sm["ST"] = sm.STATE_CODE.fillna(sm.PRIMACY_AGENCY_CODE)      # system's own state (primacy codes 01-10 are EPA regions)
    st = sm.groupby("ST").agg(N=("PWSID", "size"), people=("P", "sum"), mean_f=("f", "mean"),
                                              pf=("f", lambda s: np.sum(s * sm.loc[s.index, "P"]))).reset_index()
    g = sm.groupby("ST")
    for nm in ["p", "p_lo", "p_hi"]:
        st[nm.replace("p", "prev")] = g[nm].mean().values
        st[nm.replace("p", "people_exc")] = g.apply(lambda d, nm=nm: np.sum(d[nm] * d.P)).values
    st["systems"] = st.prev * st.N; st["prev_ratio_k"] = k * st.mean_f
    st.assign(k=k, k_lo=klo, k_hi=khi).to_csv(OUT / "fig5_state_ratio_estimates.csv", index=False)
    pd.read_csv(R / "small_burden_kg.csv").to_csv(OUT / "fig5_burden.csv", index=False)


def fig6():
    sm = pd.read_parquet(D / "small_frame_anchor.parquet"); lab = sm[sm.y_mcl.notna()]
    curves = [pd.DataFrame({"set": "UCMR5 small sample (national)", "y": lab.y_mcl.astype(int).values, "f": lab.f.values})]
    for st, (truth, e) in {"Michigan": (ext.load_michigan(), "y_avg"), "Minnesota": (ext.load_minnesota(), "y_max"),
                           "New Hampshire": (ext.load_newhampshire(), "y_fed")}.items():
        code = {"Michigan": "MI", "Minnesota": "MN", "New Hampshire": "NH"}[st]
        m = sm[(sm.PRIMACY_AGENCY_CODE == code) & sm.y_mcl.isna()].merge(truth, on="PWSID")
        curves.append(pd.DataFrame({"set": st, "y": m[e].values.astype(int), "f": m.f.values}))
    rows = []
    for c in curves:
        for s, d in c.groupby("set"):
            o = np.argsort(-d.f.values); yy = d.y.values[o]
            frac = np.arange(1, len(yy) + 1) / len(yy); rec = np.cumsum(yy) / yy.sum()
            keep = np.unique(np.r_[0, np.searchsorted(frac, np.linspace(0, 1, 101)[1:]).clip(max=len(yy) - 1)])
            rows.append(pd.DataFrame({"set": s, "tested": np.r_[0, frac[keep]], "found": np.r_[0, rec[keep]]}))
    pd.concat(rows).to_csv(OUT / "fig6_recall_curves.csv", index=False)
    for f in ["external_validation_systems.csv", "external_validation_followup.csv"]:
        pd.read_csv(R / f).to_csv(OUT / f"fig6_{f}", index=False)
    # AUC with bootstrap intervals: UCMR5 small sample (reference) and the three states (unmonitored small systems)
    sysv = pd.read_csv(R / "external_validation_systems.csv")
    lo, hi = ext.boot_ci(roc_auc_score, lab.y_mcl.astype(int).values, lab.f.values)
    rows = [dict(dataset="UCMR5 small sample (national)", auc=roc_auc_score(lab.y_mcl, lab.f), lo=lo, hi=hi, n=len(lab), positives=int(lab.y_mcl.sum()))]
    for st, name, e in [("MI", "Michigan", "y_avg"), ("MN", "Minnesota", "y_max"), ("NH", "New Hampshire", "y_fed")]:
        r = sysv[(sysv.state == st) & (sysv.subset == "unmonitored by UCMR5") & (sysv.endpoint == e)].iloc[0]
        rows.append(dict(dataset=name, auc=r.auc, lo=r.auc_lo, hi=r.auc_hi, n=int(r.n), positives=int(r.positives)))
    pd.DataFrame(rows).to_csv(OUT / "fig6_auc.csv", index=False)


def fig3(n=800):
    """Inclusion probabilities on the real small frame for n systems: EPA cell allocation, ANCHOR (systems), dual design."""
    sm = pd.read_parquet(D / "small_frame_anchor.parquet")
    cells = sm.cell.astype("category").cat.codes.values; obs = np.bincount(cells[sm.y_mcl.notna().values]).astype(float)
    Nh = np.bincount(cells).astype(float); u, P = sm.u.values, sm.P.values
    pi_epa = np.minimum(n * (obs / obs.sum()) / Nh, 1)[cells]
    pi_anchor = ac.cap_probs(ac.mix_scores(u, np.ones(len(u)), .5), n)
    s = .5 * (.25 * u / u.mean() + .75 * P * u / (P * u).mean()) + .5 * (.25 + .75 * P / P.mean())   # same dual score as s24
    pd.DataFrame({"PWSID": sm.PWSID, "f": sm.f, "u": u, "P": P, "scls": sm.scls, "pi_epa": pi_epa, "pi_anchor": pi_anchor,
                  "pi_dual": ac.cap_probs(s, n)}).to_csv(OUT / "fig3_inclusion_probabilities.csv", index=False)


def fig7():
    pd.read_csv(R / "within_system_rules_summary.csv").to_csv(OUT / "fig7_within_rules.csv", index=False)
    A = pd.read_csv(R / "two_level_census.csv"); key = ["population", "seed", "estimand", "n0"]
    A = A.join(A[A.strategy == "Equal-probability Poisson · full · HT"].set_index(key).mse.rename("mse_ref"), on=key); A["eff"] = A.mse_ref / A.mse
    A[A.n0 == 300].groupby(["population", "estimand", "strategy"]).agg(eff=("eff", "mean"), eff_sd=("eff", "std"), bias=("bias", "mean"),
                                                         cov_min=("cov_kg", "min"), cov_mean=("cov_kg", "mean")).reset_index().to_csv(OUT / "fig7_census.csv", index=False)
    A.groupby(["population", "estimand", "strategy", "n0"]).cov_kg.mean().reset_index().to_csv(OUT / "fig7_coverage_by_n.csv", index=False)
    B = pd.read_csv(R / "two_level_small.csv")
    ref = B[B.strategy.str.startswith("REFERENCE")].set_index(["scenario", "seed", "estimand"])[["mse", "cost_mean"]].rename(columns={"mse": "mse_ref", "cost_mean": "cost_ref"})
    B = B[~B.strategy.str.startswith("REFERENCE")].join(ref, on=["scenario", "seed", "estimand"])
    B["samples_needed"] = B.cost_mean * B.mse / B.mse_ref          # MSE taken as inversely proportional to the sample budget
    B.groupby(["scenario", "estimand", "strategy"]).agg(samples_needed=("samples_needed", "mean"), cost_ref=("cost_ref", "mean"),
                                                        cov_min=("cov_kg", "min")).reset_index().to_csv(OUT / "fig7_small.csv", index=False)


if __name__ == "__main__":
    if "--only-fig3-7" in sys.argv:
        fig3(); fig7(); print("fig3, fig7 data"); sys.exit()
    T, X = build_table()
    if "--only-fig5-6" not in sys.argv:
        fig1(T.copy()); print("fig1 data", flush=True)
        fig2(T, X); print("fig2 data", flush=True)
    for f in ["grid_summary.csv", "grid_stop_conditions.csv"]:
        pd.read_csv(R / f).to_csv(OUT / f"fig4_{f}", index=False)
    fig5(); print("fig5 data", flush=True)
    fig6(); print("fig6 data", flush=True)
    fig3(); print("fig3 data", flush=True)
    fig7(); print("fig7 data", flush=True)
    for p in sorted(OUT.glob("*.csv")):
        print(p.name, len(pd.read_csv(p)))
