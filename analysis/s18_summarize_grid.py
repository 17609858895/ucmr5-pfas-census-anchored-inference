"""Summarise the census-truth grid: relative efficiency, stop-condition checks, interval coverage, ablations, Pareto front.

Relative efficiency = MSE(SRS + Hajek) / MSE(method), computed within population x prior x seed x estimand x budget,
then averaged over seeds (sd across seeds reported). Writes results/grid_*.csv and results/grid_report.md.
"""
import sys, numpy as np, pandas as pd
from common import *
import anchor_core as ac

tag = "_quick" if "--quick" in sys.argv else ""
C = pd.read_parquet(R / f"grid_cheap{tag}.parquet")
G = pd.read_parquet(R / f"grid_gregml{tag}.parquet") if (R / f"grid_gregml{tag}.parquet").exists() else pd.DataFrame()
A = pd.concat([C, G], ignore_index=True)
A["mse"] = A.rmse ** 2
key = ["population", "prior", "seed", "estimand", "n"]
ref = A[(A.design == "SRS") & (A.estimator == "hajek")].set_index(key).mse.rename("mse_ref")
A = A.join(ref, on=key)
# GREG-ML ran only with the geo+nb prior; under SRS it does not use the prior, so its reference is the geo+nb one
A["rel_eff"] = A.mse_ref / A.mse
A["method"] = A.design + " + " + A.estimator
LABEL = {"SRS + hajek": "SRS + Hájek (classical)", "SRS + ppi": "SRS + PPI++", "SRS + greg_lin": "SRS + GREG-lin", "SRS + greg_ml": "SRS + GREG-ML",
         "PPS-pop + hajek": "PPS-pop + Hájek (EPA-like)", "PPS-pop + ppi": "PPS-pop + PPI++",
         "Neyman-H5 + hajek": "Neyman-H5 + HT (classical)", "Neyman-H10 + hajek": "Neyman-H10 + HT (classical)",
         "Neyman-H5 + greg_lin": "Neyman-H5 + GREG-lin", "Neyman-H5 + greg_ml": "Neyman-H5 + GREG-ML",
         "Neyman-H5 + ppi": "Neyman-H5 + PPI++ (hybrid)", "Neyman-H10 + ppi": "Neyman-H10 + PPI++ (hybrid)",
         "ANCHOR + ppi": "ANCHOR (Poisson + PPI++)", "ANCHOR + hajek": "ANCHOR design + Hájek", "ANCHOR + greg_lin": "ANCHOR design + GREG-lin",
         "ANCHOR + greg_ml": "ANCHOR design + GREG-ML", "ANCHOR + ppi_lam1": "ANCHOR, lambda = 1", "ANCHOR-sys + ppi": "ANCHOR systematic (fixed n)",
         "ANCHOR-sd + ppi": "ANCHOR, sd rule", "ANCHOR-tau0.25 + ppi": "ANCHOR, tau = 0.25", "ANCHOR-tau0.75 + ppi": "ANCHOR, tau = 0.75"}
A["label"] = A.method.map(LABEL).fillna(A.method)
A.to_csv(R / f"grid_all{tag}.csv", index=False)
agg = A.groupby(["population", "prior", "estimand", "n", "label"]).agg(rel_eff=("rel_eff", "mean"), rel_eff_sd=("rel_eff", "std"),
                                                                       bias=("bias", "mean"), rmse=("rmse", "mean"), seeds=("seed", "nunique"),
                                                                       **{f"cov_{iv}": (f"cov_{iv}", "mean") for iv in ac.INTERVALS},
                                                                       **{f"wid_{iv}": (f"wid_{iv}", "mean") for iv in ac.INTERVALS}).reset_index()
agg.to_csv(R / f"grid_summary{tag}.csv", index=False)
lines = ["# Census-truth grid report", "", f"Draws per configuration: cheap estimators {int(C.draws.max())} x {C.seed.nunique()} seeds; GREG-ML {int(G.draws.max()) if len(G) else 0} x {G.seed.nunique() if len(G) else 0} seeds.", ""]
main = agg[agg.prior == "geo+nb"]
for pop in ["medium", "large"]:
    for est in ["systems", "people"]:
        t = main[(main.population == pop) & (main.estimand == est)].pivot_table(index="label", columns="n", values="rel_eff").round(2)
        t = t.loc[[l for l in LABEL.values() if l in t.index]]
        lines += [f"## Relative efficiency vs SRS + Hájek — population {pop}, estimand {est} (prior SDWIS+geo+neighbour)", "", t.to_markdown(), ""]
# ---- stop-condition checks
anchor = main[main.label == "ANCHOR (Poisson + PPI++)"].set_index(["population", "estimand", "n"]).rel_eff
rows = []
for lab in ["Neyman-H5 + HT (classical)", "Neyman-H10 + HT (classical)", "PPS-pop + Hájek (EPA-like)", "SRS + GREG-lin", "SRS + GREG-ML",
            "Neyman-H5 + GREG-lin", "Neyman-H5 + GREG-ML", "Neyman-H10 + PPI++ (hybrid)", "ANCHOR design + GREG-ML"]:
    s = main[main.label == lab].set_index(["population", "estimand", "n"]).rel_eff
    r = (s / anchor).dropna()
    rows.append(dict(competitor=lab, median_ratio=r.median(), max_ratio=r.max(), share_ge_0p9=(r >= 0.9).mean(), configs=len(r)))
stop = pd.DataFrame(rows).round(3); stop.to_csv(R / f"grid_stop_conditions{tag}.csv", index=False)
lines += ["## Stop-condition check: competitor efficiency / ANCHOR efficiency", "", "Ratio >= 0.9 in a configuration means the competitor reaches 90% of ANCHOR there.", "", stop.to_markdown(index=False), ""]
# ---- intervals
cov = agg.assign(family=agg.label).groupby("family")[[f"cov_{iv}" for iv in ac.INTERVALS]].agg(["mean", "min"]).round(3)
cov.columns = [f"{a}_{b}" for a, b in cov.columns]
cov.to_csv(R / f"grid_coverage{tag}.csv")
lines += ["## Interval coverage (nominal 0.95), mean and minimum over populations x priors x estimands x budgets", "", cov.to_markdown(), ""]
w = agg.groupby("n")[[f"wid_{iv}" for iv in ac.INTERVALS]].mean()
w = w.div(w["wid_wald"], axis=0).round(3); lines += ["## Mean interval width relative to Wald", "", w.to_markdown(), ""]
# ---- prior ablation and Pareto
pa = agg[agg.label.isin(["ANCHOR (Poisson + PPI++)", "SRS + PPI++", "Neyman-H10 + HT (classical)"])].pivot_table(index=["population", "estimand", "label"], columns=["prior"], values="rel_eff", aggfunc="mean").round(2)
lines += ["## Prior ablation (mean over budgets)", "", pa.to_markdown(), ""]
par = agg[agg.label.str.startswith("dual-rho")].pivot_table(index=["population", "prior", "n", "label"], columns="estimand", values="rel_eff").round(2)
lines += ["## Pareto: one dual-purpose design evaluated on both estimands", "", par.to_markdown(), ""]
(R / f"grid_report{tag}.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
