"""Third-round figure modules, using frozen figure-data values.

These builders do not fit models, recompute simulations, or modify results.
They may be imported by the main figure entry point without a circular import.
Run this module directly only to create internal draft previews.
"""
from pathlib import Path

from f_common import *
import figstyle as fs
import f04_fig4_census_validation as f4
import f05_fig5_small_burden as f5
import f06_fig6_external_validation as f6
import f07_fig7_next_round_design as f7
import f08_supporting_figures as si


def _style(ax, xlabel=None, ylabel=None):
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, tick_size=10.5, label_size=11.5)
    ax.tick_params(length=3.4, width=.8, pad=3)
    return ax


def size_forest3(fig, ss, ctx):
    """Fig. 5d: the original four size estimates, now on horizontal rows."""
    ss = f5.own_legend(fig, ss, ctx, ("ANCHOR", "ML prediction"))
    ax = fig.add_subplot(ss)
    for yi, (label, key) in enumerate(f5.SIZE_KEY.items()):
        a = f5.B[(f5.B.group == key) & (f5.B.estimand == "systems")
                 & (f5.B.estimator == "ANCHOR PPI++")].iloc[0]
        p = f5.B[(f5.B.group == key) & (f5.B.estimand == "systems")
                 & f5.B.estimator.str.startswith("ML plug-in")].iloc[0]
        ax.plot([p.theta * 100, a.theta * 100], [yi + .09, yi - .09], color=MUTED,
                lw=.85, zorder=1)
        ax.errorbar(a.theta * 100, yi - .09,
                    xerr=[[(a.theta - a.kg_lo) * 100], [(a.kg_hi - a.theta) * 100]],
                    fmt="D", ms=5.5, mfc=C["ANCHOR (Poisson + PPI++)"],
                    mec=INK, mew=.5, ecolor=INK, elinewidth=.8, capsize=2,
                    zorder=3)
        ax.plot(p.theta * 100, yi + .09, ls="", marker="s", ms=5,
                color=C["plugin"], mec=INK, mew=.5, zorder=3)
    ax.set_yticks(range(4), list(f5.SIZE_KEY))
    ax.set_ylim(3.5, -.5)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    _style(ax, "Exceeding systems (%)", "People served per system")


def auc_forest3(fig, ss, ctx):
    """Fig. 6b: exact original AUC/CI values, with only chance as reference."""
    f6.panel_auc(fig, ss, ctx)
    ax = fig.axes[-1]
    for line in list(ax.lines):
        x = np.asarray(line.get_xdata(), dtype=float)
        if len(x) == 2 and np.allclose(x, [.7, .7]):
            line.remove()
    ax.set_xlim(.4, 1.035)
    for yi, dataset in enumerate(f6.SETS):
        r = f6.AUC[f6.AUC.dataset == dataset].iloc[0]
        ax.text(1.025, yi, f"{r.auc:.2f}", ha="right", va="center",
                fontsize=10.5)
    ax.spines["bottom"].set_bounds(.4, 1.0)
    _style(ax, "AUC")


def shared_design_summary3():
    """Pure summary: same retained rows and formula as round-two shared ranges."""
    d = si.SM[si.SM.strategy == "Dual · R12 + two-phase"].copy()
    refs = si.SM[si.SM.strategy.str.startswith("REFERENCE")].set_index(
        ["scenario", "seed", "estimand"]).mse
    d = d.join(refs.rename("mse_ref"), on=["scenario", "seed", "estimand"])
    d["need"] = d.cost_mean * d.mse / d.mse_ref
    return d.groupby(["scenario", "estimand"]).agg(
        need_mean=("need", "mean"), need_min=("need", "min"),
        need_max=("need", "max"), coverage_mean=("cov_kg", "mean"),
        coverage_min=("cov_kg", "min"), coverage_max=("cov_kg", "max")
    ).reset_index()


def projected_cost3(fig, ss, ctx):
    """Fig. 7c: all original six cost groups and both scenario markers/ranges."""
    f7.panel_small(fig, ss, ctx)
    ax = fig.axes[-1]
    ax.set_yticks(range(6), ["Cells: full", "Cells: R12 + correction",
        "ANCHOR: full", "ANCHOR: R12 + correction",
        "ANCHOR: R1 + correction", "Dual: R12 + correction"])
    ax.set_xlabel("Projected analytical samples")
    legend = ax.get_legend()
    # f7 stores the legend in its separate strip axis.
    for axis in fig.axes:
        legend = axis.get_legend()
        if legend:
            for text in legend.get_texts():
                if text.get_text() == "Higher AUC":
                    text.set_text("AUC matched")
    d = shared_design_summary3()
    for j, estimand in enumerate(["systems", "people"]):
        yy = 5 + (j - .5) * .38
        for scenario, dy in [("fitted", -.055), ("auc-match", .055)]:
            v = d[(d.scenario == scenario) & (d.estimand == estimand)].iloc[0]
            ax.errorbar(v.need_mean, yy + dy,
                        xerr=[[v.need_mean - v.need_min],
                              [v.need_max - v.need_mean]],
                        fmt="none", ecolor=INK, elinewidth=.8, capsize=2,
                        zorder=4)
    _style(ax, "Projected analytical samples")


def shared_coverage3(fig, ss, ctx):
    """Fig. 7e: four original coverage means/ranges in paired scenario rows."""
    g = ss.subgridspec(2, 1, height_ratios=[.18, 1], hspace=.035)
    handles = [Line2D([], [], ls="", marker=marker, mfc=C[estimand],
                     mec=INK, mew=.5, ms=5.5, label=label)
               for estimand, marker, label in [
                   ("systems", "s", "Systems"),
                   ("people", "D", "People served")]]
    legend_strip(fig, g[0], handles, 2, columnspacing=.8, handletextpad=.25)
    ax = fig.add_subplot(g[1])
    d = shared_design_summary3()
    for row, scenario in enumerate(["fitted", "auc-match"]):
        for estimand, marker, offset in [("systems", "s", -.13),
                                         ("people", "D", .13)]:
            r = d[(d.scenario == scenario) & (d.estimand == estimand)].iloc[0]
            ax.errorbar(r.coverage_mean, row + offset,
                        xerr=[[r.coverage_mean - r.coverage_min],
                              [r.coverage_max - r.coverage_mean]],
                        fmt=marker, ms=5.5, mfc=C[estimand], mec=INK,
                        mew=.5, ecolor=INK, elinewidth=.9, capsize=2,
                        zorder=3)
    ax.set_yticks([0, 1], ["Fitted", "AUC matched"])
    ax.set_ylim(1.5, -.5)
    ax.set_xlim(.915, .965)
    ax.set_xticks([.92, .94, .96])
    ax.axvline(.95, color=MUTED, ls="--", lw=.8, zorder=1)
    _style(ax, "95% interval coverage")


def depth_coverage3(fig, ss, ctx):
    """Fig. 7d: frozen depth coverage with a safe three-row legend strip."""
    handles = [Line2D([], [], color=f7.RC[rule], lw=1.5, ls=linestyle,
                      marker=marker, ms=4,
                      mfc=f7.RC[rule] if linestyle == "-" else "white",
                      mec=INK, mew=.5, label=label)
               for _, label, rule, linestyle, marker in f7.C_LINES]
    g = ss.subgridspec(2, 1, height_ratios=[.5, 1], hspace=.07)
    legend_strip(fig, g[0], handles, 2, handlelength=1.15,
                 columnspacing=.75, labelspacing=.25)
    ax = fig.add_subplot(g[1])
    ns = [150, 300, 600]
    for j, (strategy, _, rule, linestyle, marker) in enumerate(f7.C_LINES):
        values = f7.CV[f7.CV.strategy == strategy].groupby("n0").cov_kg
        x = np.arange(3) + (j - 2) * .07
        mean = values.mean().reindex(ns)
        lo, hi = values.min().reindex(ns), values.max().reindex(ns)
        ax.vlines(x, lo, hi, color=f7.RC[rule], lw=1)
        ax.plot(x, mean, color=f7.RC[rule], lw=1.5, ls=linestyle,
                marker=marker, ms=4.5,
                mfc=f7.RC[rule] if linestyle == "-" else "white",
                mec=INK, mew=.45)
    ax.axhline(.95, color=INK, lw=.7, ls=":")
    ax.set_ylim(.4, .98)
    ax.set_yticks([.4, .6, .8, .95])
    ax.set_xlim(-.35, 2.35)
    ax.set_xticks(range(3), ns)
    _style(ax, "Budget (system equivalents)", "95% interval coverage")


def layout7(fig):
    """a,b upper row; c full-width projected costs; d,e lower row."""
    g = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.03, .9],
                         left=.065, right=.99, top=.99, bottom=.045,
                         hspace=.12)
    top = g[0].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=.16)
    low = g[2].subgridspec(1, 2, width_ratios=[1.13, 1], wspace=.18)
    return [top[0], top[1], g[1], low[0], low[1]]


def ablation_dots3(fig, ss, ctx):
    """Fig. S2e: same nine design variants and 18 efficiencies as paired dots."""
    g = ss.subgridspec(2, 1, height_ratios=[.22, 1], hspace=.04)
    handles = [Line2D([], [], ls="", marker=marker, mfc=C[estimand],
                     mec=INK, mew=.5, ms=5.5, label=label)
               for estimand, marker, label in [("systems", "s", "Systems"),
                                               ("people", "D", "People")]]
    legend_strip(fig, g[0], handles, 2)
    ax = fig.add_subplot(g[1])
    labels = ["ANCHOR", "τ = 0.25", "τ = 0.75", "λ = 1", "Fixed n",
              "SD score", "GREG-lin", "Hájek", "Uniform"]
    for yi, (method, _) in enumerate(f4.ABL):
        values = [f4.G[(f4.G.population == "medium") & (f4.G.n == 300)
                       & (f4.G.estimand == estimand)
                       & (f4.G.label == method)].rel_eff.iloc[0]
                  for estimand in ["systems", "people"]]
        ax.plot(values, [yi, yi], color=MUTED, lw=.9, zorder=1)
        for value, estimand, marker in zip(values, ["systems", "people"],
                                          ["s", "D"]):
            ax.plot(value, yi, ls="", marker=marker, ms=5.5,
                    mfc=C[estimand], mec=INK, mew=.5, zorder=3)
    ax.axvline(1, color=MUTED, lw=.8, ls="--", zorder=1)
    ax.set_yticks(range(9), labels)
    ax.set_ylim(8.5, -.5)
    ax.set_xlim(.9, 2.6)
    ax.set_xticks([1, 1.5, 2, 2.5])
    _style(ax, "Efficiency vs SRS")


def _preview():
    """Internal previews only; the main entry point remains authoritative."""
    import generate_refined_figures as base
    from publication_layout import draw_module
    out = OUTPUT / "checks" / "第三轮草稿"
    out.mkdir(parents=True, exist_ok=True)
    fs.QA = out / "qa"
    plans = [
        ("Fig5_round3", [f5.panel_map, base.states_lollipop,
                         f5.panel_national, size_forest3], base.layout5, (190, 143)),
        ("Fig6_round3", [base.prevalence_facets, auc_forest3, base.recall],
         base.layout6, (190, 137)),
        ("Fig7_round3", [base.rules, base.depth_eff, projected_cost3,
                         depth_coverage3, shared_coverage3], layout7, (190, 190)),
        ("FigS2_round3", [base.sensitivity_heatmap("systems"),
                           base.sensitivity_heatmap("people"),
                           base.shared_panel(base.feature_dumbbell("systems")),
                           base.shared_panel(base.feature_dumbbell("people")),
                           ablation_dots3], base.supplement2, (190, 205)),
    ]
    for stem, builders, layout, size in plans:
        fig = plt.figure(figsize=np.array(size) / 25.4)
        for builder, ss, letter in zip(builders, layout(fig), "abcde"):
            draw_module(fig, ss, letter, base.polish(builder))
        report = fs._export(fig, out, stem, True)
        print(stem, report["alignment_max_residual_pt"],
              report["n_unexpected"], len(report["label_overlaps"]), flush=True)


if __name__ == "__main__":
    _preview()
