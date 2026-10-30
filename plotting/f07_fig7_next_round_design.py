"""Fig. 7 — Next-round small-system design: sampling depth within systems and the two-level analytical-sample budget."""
from f_common import *

W = pd.read_csv(FD / "fig7_within_rules.csv"); E = pd.read_csv(FD / "fig7_census.csv")
CV = pd.read_csv(FD / "fig7_coverage_by_n.csv"); S = pd.read_csv(FD / "fig7_small.csv")
# one colour per depth rule (panels a, c); estimand colours (panels b, d) follow C["systems"], C["people"]
RC = {"full": PAL[1], "R2": PAL[2], "R12": PAL[6], "R1": PAL[0]}
GROUPS = [("small (UCMR5 sample)", "Small", "o"), ("medium census", "Medium", "s"), ("large census", "Large", "^")]
EST = [("systems", "Systems"), ("people", "People")]


def with_legend(fig, ss, handles, ncol, h=.16, loc="center right"):
    """Legend strip above the data axes; right-anchored so a wide legend overhangs the tick-label column, not the right edge."""
    gs = ss.subgridspec(2, 1, height_ratios=[h, 1], hspace=.04)
    la = fig.add_subplot(gs[0]); la.axis("off")
    la.legend(handles=handles, loc=loc, ncol=ncol, handlelength=.65,handletextpad=.3, columnspacing=.55, borderaxespad=0)
    return fig.add_subplot(gs[1])


def est_handles(extra=()):
    return [Patch(fc=C[k], ec=INK, lw=.5, label=l) for k, l in EST] + list(extra)


def panel_rules(fig, ss, ctx):
    ax = with_legend(fig, ss, [Line2D([], [], ls="", marker=m, ms=5, mfc="white", mec=INK, mew=.6, label=l) for _, l, m in GROUPS], 3, loc="center")
    for g, _, m in GROUPS:
        for rule in ["R2", "R12", "R1"]:
            r = W[(W.group == g) & (W.rule == rule)].iloc[0]
            ax.plot(r.cost_saved * 100, r.miss_rate * 100, ls="", marker=m, ms=5.5, color=RC[rule], mec=INK, mew=.5, zorder=3)
    for rule, (x, y, ha) in {"R2": (17.5, 1.5, "center"), "R12": (42.5, 2.4, "right"), "R1": (52, 5.4, "right")}.items():
        ax.text(x, y, rule, ha=ha, va="center", fontsize=8.5)
    ax.set_xlim(0, 62); ax.set_xticks([0, 20, 40, 60]); ax.set_ylim(-.3, 7.5); ax.set_yticks([0, 2, 4, 6])
    style_axis(ax, xlabel="Analytical samples saved (%)", ylabel="Exceeding systems missed (%)")


B_ROWS = [("Equal-probability Poisson · R12 · HT", "Equal π, R12", True), ("ANCHOR · full", "ANCHOR, full", False), ("ANCHOR · R12", "ANCHOR, R12", True),
          ("ANCHOR · R12 + two-phase q=0.2", "ANCHOR, R12 + 2-phase", False), ("ANCHOR · R1", "ANCHOR, R1", True),
          ("ANCHOR · R1 + two-phase q=0.2", "ANCHOR, R1 + 2-phase", False)]


def panel_eff(fig, ss, ctx):
    ax = with_legend(fig, ss, est_handles([Patch(fc="white", ec=INK, lw=.5, hatch="////", label="Uncorrected")]), 3)
    d = E[E.population == "medium"]; y = np.arange(len(B_ROWS))
    for k, (key, _) in enumerate(EST):
        for yi, (strat, _, unc) in zip(y, B_ROWS):
            r = d[(d.estimand == key) & (d.strategy == strat)].iloc[0]; yy = yi + (k - .5) * .38
            ax.barh(yy, r.eff, height=.36, color=C[key], edgecolor=INK, lw=.45, hatch="////" if unc else None, zorder=2)
            ax.errorbar(r.eff, yy, xerr=r.eff_sd, fmt="none", ecolor=INK, elinewidth=.7, capsize=1.5, zorder=3)
    ax.axvline(1, color=INK, lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, [r[1] for r in B_ROWS]); ax.set_ylim(len(y) - .45, -.55)
    ax.set_xlim(0, 5); ax.set_xticks([0, 1, 2, 3, 4, 5])
    style_axis(ax, xlabel="Efficiency vs equal π, full")
    plt.setp(ax.get_yticklabels(), fontsize=8.5)


C_LINES = [("ANCHOR · full", "Full", "full", "-", "o"), ("ANCHOR · R12", "R12", "R12", (0, (3, 1.6)), "s"),
           ("ANCHOR · R12 + two-phase q=0.2", "R12 + 2-phase", "R12", "-", "s"), ("ANCHOR · R1", "R1", "R1", (0, (3, 1.6)), "D"),
           ("ANCHOR · R1 + two-phase q=0.2", "R1 + 2-phase", "R1", "-", "D")]


def panel_cov(fig, ss, ctx):
    hs = [Line2D([], [], color=RC[r], lw=1.3, ls=ls, marker=m, ms=4, mfc=RC[r] if ls == "-" else "white", mec=INK, mew=.5, label=l)
          for _, l, r, ls, m in C_LINES]
    # columns: full | R12 corrected over uncorrected | R1 corrected over uncorrected
    hs = [hs[0], Line2D([], [], ls="", color=INK, label=" "), hs[2], hs[1], hs[4], hs[3]]
    ax = with_legend(fig, ss, hs, 3, h=.24)
    n0 = [150, 300, 600]
    for j, (strat, _, rule, ls, m) in enumerate(C_LINES):
        g = CV[CV.strategy == strat].groupby("n0").cov_kg; x = np.arange(3) + (j - 2) * .07
        mean, lo, hi = g.mean().reindex(n0).values, g.min().reindex(n0).values, g.max().reindex(n0).values
        ax.vlines(x, lo, hi, color=RC[rule], lw=1.1, zorder=2)
        ax.plot(x, mean, color=RC[rule], lw=1.3, ls=ls, marker=m, ms=4.2, mfc=RC[rule] if ls == "-" else "white", mec=INK, mew=.5, zorder=3)
    ax.axhline(.95, color=INK, lw=.7, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xticks(range(3), [str(v) for v in n0]); ax.set_xlim(-.35, 2.35)
    ax.set_ylim(.4, .98); ax.set_yticks([.4, .6, .8, .95])
    style_axis(ax, xlabel="Budget (full-schedule systems)", ylabel="95% interval coverage")


D_ROWS = [("EPA allocation · full", "Cell allocation, full"), ("EPA allocation · R12 + two-phase", "Cell allocation, R12 + 2-phase"), ("ANCHOR · full", "ANCHOR, full"),
          ("ANCHOR · R12 + two-phase", "ANCHOR, R12 + 2-phase"), ("ANCHOR · R1 + two-phase", "ANCHOR, R1 + 2-phase"),
          ("Dual · R12 + two-phase", "Dual, R12 + 2-phase")]


def panel_small(fig, ss, ctx):
    ax = with_legend(fig, ss, est_handles([Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=INK, mew=.6, label="Higher AUC")]), 3)
    y = np.arange(len(D_ROWS))
    for k, (key, _) in enumerate(EST):
        for yi, (strat, _) in zip(y, D_ROWS):
            f = S[(S.scenario == "fitted") & (S.estimand == key) & (S.strategy == strat)].iloc[0]
            a = S[(S.scenario == "auc-match") & (S.estimand == key) & (S.strategy == strat)].iloc[0]
            yy = yi + (k - .5) * .38
            ax.barh(yy, f.samples_needed, height=.36, color=C[key], edgecolor=INK, lw=.45, zorder=2)
            ax.plot(a.samples_needed, yy, ls="", marker="D", ms=3.6, mfc="white", mec=INK, mew=.6, zorder=3)
    ref = S[S.scenario == "fitted"].cost_ref.mean()
    ax.axvline(ref, color=INK, lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, [r[1] for r in D_ROWS]); ax.set_ylim(len(y) - .45, -.55)
    ax.set_xlim(0, 3200); ax.set_xticks([0, 1000, 2000, 3000], ["0", "1,000", "2,000", "3,000"])
    style_axis(ax, xlabel="Projected samples for equal MSE")
    plt.setp(ax.get_yticklabels(), fontsize=8.5)


def layout(fig):
    g = fig.add_gridspec(2, 1, left=.035, right=.985, top=.99, bottom=.06, hspace=.14)
    top = g[0].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=.12)
    low = g[1].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=.12)
    return [top[0], top[1], low[0], low[1]]


if __name__ == "__main__":
    save_modules([panel_rules, panel_eff, panel_cov, panel_small], "Fig7_next_round_design", "Fig7_next_round_design", layout,
                 figsize_mm=(180, 140), sizes_mm=[(88, 68), (95, 68), (88, 70), (95, 68)])
