"""Same-depth classical control: target-specific five-stratum Neyman allocation with R12 schedules and 20% random completion.

Complements external_classical_controls.py, whose Neyman references use full schedules only. Strata, stratum standard
deviations and integer plans follow neyman_specs there, except that each stratum's expected unit cost is the two-phase
cost q*c_full + (1-q)*c_R12. Adjacent integer plans are mixed at random to match the expected budget exactly. Point
estimates use the same two-phase difference estimator as the other designs (estimator_core.estimate, power-tuned lambda);
its variance uses the Poisson approximation with pi = n_h/N_h, which ignores the stratification gain and is therefore
conservative for stratified selection. No external outcome is used to form strata or allocations.

Usage: python -m strengthening.method.external_neyman_two_phase --cohort <external_schedule_systems.csv>
                                                                --primary <external_primary_R12_B300.csv> --out <dir>
"""
from pathlib import Path
import argparse, json, hashlib, sys, time
import numpy as np
import pandas as pd
from scipy.stats import skew

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from strengthening import paths  # noqa: E402
sys.path.insert(0, str(paths.ANALYSIS_CODE))
import anchor_core as ac  # noqa: E402
from strengthening.method.estimator_core import estimate  # noqa: E402

R = 1000; Q = .2; H = 5; BUDGET_EQ = [100, 300]


def summarize(rows, truth):
    """Same summary fields as completion_design_validation.summarize (MSE, bias, coverage of all interval variants)."""
    x = np.array(rows); err = x[:, 0] - truth; iv = ac.intervals_vec(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    o = {'truth': truth, 'bias': float(err.mean()), 'bias_mc_se': float(err.std(ddof=1) / np.sqrt(len(err))),
         'mse': float(np.mean(err ** 2)), 'mse_mc_se': float(np.std(err ** 2, ddof=1) / np.sqrt(len(err))),
         'empirical_sd': float(err.std(ddof=1)), 'mean_variance': float(np.mean(x[:, 1] ** 2)),
         'variance_ratio': float(np.mean(x[:, 1] ** 2) / np.var(err, ddof=1)), 'error_skewness': float(skew(err)),
         'mean_n': float(x[:, 3].mean()), 'lambda_mean': float(x[:, 4].mean()), 'mean_full_n': float(x[:, 5].mean()),
         'zero_missed_full_fraction': float(np.mean(x[:, 6] == 0))}
    for name, (lo, hi) in iv.items():
        o['coverage_' + name] = float(np.mean((lo <= truth) & (truth <= hi))); o['width_' + name] = float(np.mean(hi - lo))
    o['coverage_kg_mc_se'] = float(np.sqrt(o['coverage_kg'] * (1 - o['coverage_kg']) / len(err)))
    return o


def neyman_plans(f, a, unit_cost, B):
    p = np.clip(f, .001, .999); st = ac.rank_strata(a * f, a * np.sqrt(p * (1 - p)), H=H)
    Nh = np.bincount(st, minlength=H).astype(float)
    Sh = np.array([np.sqrt(max(np.mean(a[st == h] ** 2 * p[st == h]) - np.mean(a[st == h] * p[st == h]) ** 2, 1e-12)) for h in range(H)])
    ch = np.array([unit_cost[st == h].mean() for h in range(H)])
    plans = [(n, float(np.sum(nh * ch)), nh) for n in range(2 * H, len(f) + 1) for nh in [ac.neyman_alloc(Nh, Sh, n)]]
    below = [x for x in plans if x[1] <= B]; above = [x for x in plans if x[1] >= B]
    low = max(below, key=lambda x: x[1]); high = min(above, key=lambda x: x[1])
    prob = 0. if np.isclose(high[1], low[1]) else (B - low[1]) / (high[1] - low[1])
    return st, Nh, low, high, prob


def run_state(g, si):
    y = g.y_full.to_numpy(float); yt = g['y_R12'].to_numpy(float); f = g.f.to_numpy(float); P = g.P.to_numpy(float)
    c_full = g.c_full.to_numpy(float); c_r12 = g['c_R12'].to_numpy(float); N = len(y); q = np.full(N, Q)
    unit = Q * c_full + (1 - Q) * c_r12; out = []
    for b0 in BUDGET_EQ:
        B = b0 * c_full.mean()
        for ti, (target, a) in enumerate([('systems', np.ones(N)), ('people', P)]):
            st, Nh, low, high, prob = neyman_plans(f, a, unit, B)
            rng = np.random.default_rng(990000 + si * 100 + ti * 10 + int(B)); rows = []; costs = []
            truth = float(np.sum(a * y) / a.sum())
            for _ in range(R):
                nh = high[2] if rng.random() < prob else low[2]
                I = ac.draw_stratified(st, nh, rng); pi = (nh / Nh)[st]; J = rng.random(N) < q
                costs.append(float(np.where(J[I], c_full[I], c_r12[I]).sum()))
                rows.append(estimate(y, yt, f, a, pi, I, J, q, 'tuned'))
            out.append({'population': g.external_state.iloc[0], 'rule': 'R12', 'budget_full_equivalents': b0, 'budget': B,
                        'policy': 'neyman5_two_phase_difference', 'estimand': target, 'draws': R, 'q': Q,
                        'expected_cost': float((1 - prob) * low[1] + prob * high[1]), 'mean_cost': float(np.mean(costs)),
                        'N': N, 'positives': int(y.sum()), 'missed_full_population': int((y - yt).sum()), **summarize(rows, truth)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, required=True); ap.add_argument('--primary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    data = pd.read_csv(a.cohort, dtype={'PWSID': str})
    out = []
    for si, (state, g) in enumerate(data.groupby('external_state', sort=True)):
        out += run_state(g, si)
    res = pd.DataFrame(out)
    prim = pd.read_csv(a.primary); prim = prim[~prim.coefficient.eq('one')]
    ref = prim[prim.policy.eq('srs_full_hajek')].set_index(['population', 'estimand']).mse
    ney = prim[prim.policy.eq('neyman5_full_difference')].set_index(['population', 'estimand']).mse
    b300 = res.budget_full_equivalents.eq(300)
    res.loc[b300, 'mse_ratio_to_SRS_Hajek'] = [r.mse / ref[(r.population, r.estimand)] for r in res[b300].itertuples()]
    res.loc[b300, 'mse_ratio_to_Neyman_full'] = [r.mse / ney[(r.population, r.estimand)] for r in res[b300].itertuples()]
    res.to_csv(a.out / 'external_neyman_two_phase.csv', index=False)
    (a.out / 'external_neyman_two_phase_run.json').write_text(json.dumps({
        'cohort_sha256': hashlib.sha256(a.cohort.read_bytes()).hexdigest(), 'draws': R, 'q': Q, 'strata': H,
        'budgets_full_equivalents': BUDGET_EQ, 'allocation_uses_external_outcomes': False,
        'variance': 'Poisson approximation with pi=n_h/N_h (conservative for stratified selection)',
        'elapsed_seconds': round(time.perf_counter() - t0, 1)}, indent=2), encoding='utf-8')
    print(res[['population', 'budget_full_equivalents', 'estimand', 'mse_ratio_to_SRS_Hajek', 'mse_ratio_to_Neyman_full',
               'coverage_kg', 'mean_n', 'mean_cost']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
