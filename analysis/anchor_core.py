"""ANCHOR core: finite-population designs, estimators and intervals for a (weighted) prevalence.

Target: theta = sum_U a_i y_i / sum_U a_i, with a_i = 1 (share of systems) or a_i = population served (share of people).
Every estimator returns (theta_hat, se, df). Variances are design-based:
  Poisson-type designs : V = sum_s (1 - pi_i) (w_i z_i)^2 / A^2
  stratified SRSWOR    : V = sum_h N_h^2 (1 - n_h/N_h) s_h^2(z) / n_h / A^2   (SRS is one stratum)
where z_i is the linearised (or residual) variable of unit i and A = sum_U a_i.
"""
import numpy as np
from scipy.stats import norm, t as tdist, beta
from sklearn.linear_model import Ridge
import lightgbm as lgb

Z = norm.ppf(0.975)


# ----------------------------------------------------------------------------- designs
def cap_probs(s, n):
    """Inclusion probabilities proportional to s with expected size n, iteratively capped at 1."""
    s = np.asarray(s, float).clip(min=1e-12)
    pi = np.zeros_like(s); fixed = np.zeros(len(s), bool)
    for _ in range(50):
        rest = n - fixed.sum()
        pi[~fixed] = rest * s[~fixed] / s[~fixed].sum()
        over = (pi > 1) & ~fixed
        if not over.any():
            break
        fixed |= over; pi[fixed] = 1.0
    return pi.clip(max=1.0)


def mix_scores(score, base, tau):
    """(1 - tau) * score/mean(score) + tau * base/mean(base); tau > 0 keeps every pi_i > 0."""
    return (1 - tau) * score / score.mean() + tau * base / base.mean()


def draw_poisson(pi, rng):
    return rng.random(len(pi)) < pi


def draw_systematic(pi, rng):
    """Systematic pi-ps sampling in random order: fixed size, exact inclusion probabilities."""
    order = rng.permutation(len(pi)); c = np.cumsum(pi[order]); u = rng.random()
    hit = np.floor(c - u) > np.floor(np.r_[0.0, c[:-1]] - u)
    m = np.zeros(len(pi), bool); m[order[hit]] = True
    return m


def neyman_alloc(Nh, Sh, n, nmin=2):
    """Neyman allocation n_h ~ N_h S_h, capped at N_h, at least nmin, largest-remainder rounding."""
    Nh = np.asarray(Nh, float); Sh = np.asarray(Sh, float).clip(min=1e-9)
    nh = np.minimum(np.maximum(n * Nh * Sh / np.sum(Nh * Sh), nmin), Nh)
    for _ in range(20):                       # re-spread budget after caps / floors
        free = (nh < Nh) & (nh > nmin)
        gap = n - nh.sum()
        if abs(gap) < 1e-9 or not free.any():
            break
        nh[free] += gap * (Nh[free] * Sh[free]) / np.sum(Nh[free] * Sh[free])
        nh = np.minimum(np.maximum(nh, nmin), Nh)
    base = np.floor(nh).astype(int); rem = int(round(n - base.sum()))
    if rem > 0:
        order = np.argsort(-(nh - base) + (base >= Nh) * 10)
        for j in order[:rem]:
            if base[j] < Nh[j]:
                base[j] += 1
    return base


def draw_stratified(strata, nh, rng):
    """SRSWOR of nh[h] units in each stratum h (strata coded 0..H-1)."""
    key = rng.random(len(strata))
    m = np.zeros(len(strata), bool)
    for h in range(len(nh)):
        idx = np.flatnonzero(strata == h)
        if nh[h] > 0:
            m[idx[np.argsort(key[idx])[: nh[h]]]] = True
    return m


def rank_strata(*keys, H):
    """Equal-count strata from the ordering of `keys` (last key is primary, as in np.lexsort); ties never empty a stratum."""
    order = np.lexsort(keys); strata = np.empty(len(order), int)
    strata[order] = np.arange(len(order)) * H // len(order)
    return strata


# ----------------------------------------------------------------------------- variance helpers
def _var(z, w, pi, A, design):
    """design = ('poisson',) or ('strat', strata_of_sampled, Nh, nh)."""
    if design[0] == "poisson":
        return np.sum((1 - pi) * (w * z) ** 2) / A ** 2
    _, hs, Nh, nh = design
    v = 0.0
    for h in np.unique(hs):
        zh = z[hs == h]
        if len(zh) > 1:
            v += Nh[h] ** 2 * (1 - nh[h] / Nh[h]) * zh.var(ddof=1) / nh[h]
    return v / A ** 2


def _df(n, design):
    return max(n - (len(np.unique(design[1])) if design[0] == "strat" else 1), 1)


# ----------------------------------------------------------------------------- estimators
def est_hajek(y, a, w, pi, A, design):
    th = np.sum(w * a * y) / np.sum(w * a)
    z = a * (y - th) * A / np.sum(w * a)      # linearised variable on the A scale
    return th, np.sqrt(_var(z, w, pi, A, design)), _df(len(y), design)


def _lam(y, f, a, w):
    yb = np.sum(w * a * y) / np.sum(w * a); fb = np.sum(w * a * f) / np.sum(w * a)
    return np.clip(np.sum(w * a ** 2 * (y - yb) * (f - fb)) / max(np.sum(w * a ** 2 * (f - fb) ** 2), 1e-12), 0.0, 2.0)


def est_ppi(y, f, a, w, pi, A, sumAF, design, lam=None):
    """Prediction-powered (difference / AIPW) estimator with an external prediction f; lam=None -> power-tuned."""
    lam = _lam(y, f, a, w) if lam is None else lam
    e = a * (y - lam * f)
    th = (lam * sumAF + np.sum(w * e)) / A
    return th, np.sqrt(_var(e, w, pi, A, design)), max(_df(len(y), design) - 1, 1)


def est_greg(y, a, w, pi, A, design, Xs, XU, aU, model="lin", seed=0, folds=5):
    """Model-assisted GREG: working model fitted on the SAMPLE only (design x a weights).
    lin : ridge-stabilised linear model with intercept (classical GREG);
    ml  : LightGBM (cross-entropy) cross-fitted over `folds` sample folds; sample residuals use out-of-fold fits,
          the population sum uses the average of the fold models."""
    sw = w * a
    if model == "lin":
        mu, sd = XU.mean(0), XU.std(0) + 1e-9
        r = Ridge(alpha=1.0).fit((Xs - mu) / sd, y, sample_weight=sw / sw.mean())
        mS = r.predict((Xs - mu) / sd); mU = r.predict((XU - mu) / sd)
    else:
        rng = np.random.default_rng(seed); fid = rng.integers(0, folds, len(y))
        mS = np.zeros(len(y)); mU = np.zeros(len(XU))
        for k in range(folds):
            tr = fid != k
            g = lgb.LGBMRegressor(objective="cross_entropy", n_estimators=150, learning_rate=0.05, num_leaves=7,
                                  min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  n_jobs=1, verbose=-1, random_state=seed + k)
            g.fit(Xs[tr], y[tr], sample_weight=sw[tr] / sw[tr].mean())
            mS[~tr] = g.predict(Xs[~tr]); mU += g.predict(XU) / folds
    e = a * (y - mS)
    th = (np.sum(aU * mU) + np.sum(w * e)) / A
    p = Xs.shape[1] if model == "lin" else 1
    return th, np.sqrt(_var(e, w, pi, A, design)), max(_df(len(y), design) - p, 1)


# ----------------------------------------------------------------------------- intervals
def _neff(th, se, n):
    thc = min(max(th, 0.0), 1.0)
    if se <= 0 or thc <= 0 or thc >= 1:
        return float(n)
    return thc * (1 - thc) / se ** 2


def _wilson(p, n):
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return c - h, c + h


def intervals(th, se, df, n):
    """Return dict name -> (lo, hi). 'kg' is the Korn-Graubard-type Clopper-Pearson interval on the
    design-effective sample size with a degrees-of-freedom correction (t_{n-1}/t_df)^2."""
    out = {}
    thc = min(max(th, 0.0), 1.0)
    out["wald"] = (th - Z * se, th + Z * se)
    tq = tdist.ppf(0.975, df)
    out["wald_t"] = (th - tq * se, th + tq * se)
    if 0 < th < 1 and se > 0:
        lg = np.log(th / (1 - th)); d = tq * se / (th * (1 - th))
        out["logit_t"] = (1 / (1 + np.exp(-(lg - d))), 1 / (1 + np.exp(-(lg + d))))
    else:
        out["logit_t"] = (0.0, th + tq * se)
    ne = _neff(th, se, n)
    out["wilson"] = _wilson(thc, ne)
    ne_df = ne * (Z / tq) ** 2
    out["wilson_df"] = _wilson(thc, ne_df)
    adj = (tdist.ppf(0.975, max(n - 1, 1)) / tq) ** 2
    nk = max(ne * adj, 1.0); x = thc * nk
    lo = 0.0 if x <= 0 else beta.ppf(0.025, x, nk - x + 1)
    hi = 1.0 if x >= nk else beta.ppf(0.975, x + 1, nk - x)
    out["kg"] = (lo, hi)
    return out


INTERVALS = ["wald", "wald_t", "logit_t", "wilson", "wilson_df", "kg"]


def _wilson_v(p, n):
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return c - h, c + h


def intervals_vec(th, se, df, n):
    """Vectorised version of `intervals` over arrays of draws."""
    th, se, df, n = (np.asarray(v, float) for v in (th, se, df, n))
    thc = th.clip(0, 1); tq = tdist.ppf(0.975, df); out = {"wald": (th - Z * se, th + Z * se), "wald_t": (th - tq * se, th + tq * se)}
    ok = (th > 0) & (th < 1) & (se > 0)
    lg = np.log(np.where(ok, th, 0.5) / (1 - np.where(ok, th, 0.5))); dd = tq * se / np.where(ok, th * (1 - th), 1)
    out["logit_t"] = (np.where(ok, 1 / (1 + np.exp(-(lg - dd))), 0.0), np.where(ok, 1 / (1 + np.exp(-(lg + dd))), th + tq * se))
    good = (se > 0) & (thc > 0) & (thc < 1)
    ne = np.where(good, thc * (1 - thc) / np.where(se > 0, se, 1) ** 2, n)
    out["wilson"] = _wilson_v(thc, ne)
    out["wilson_df"] = _wilson_v(thc, ne * (Z / tq) ** 2)
    nk = np.maximum(ne * (tdist.ppf(0.975, np.maximum(n - 1, 1)) / tq) ** 2, 1.0); x = thc * nk
    lo = np.where(x <= 0, 0.0, beta.ppf(0.025, np.maximum(x, 1e-9), nk - x + 1))
    hi = np.where(x >= nk, 1.0, beta.ppf(0.975, x + 1, np.maximum(nk - x, 1e-9)))
    out["kg"] = (lo, hi)
    return out
