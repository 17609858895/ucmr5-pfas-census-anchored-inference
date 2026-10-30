"""Fig. 4 — census-truth validation: efficiency, the classical baselines that catch up, dual-purpose designs, intervals."""
from f_common import *

G = pd.read_csv(FD / "fig4_grid_summary.csv"); G = G[G.prior == "geo+nb"]
METHODS = [("SRS + Hájek (classical)", "SRS", "o", "-"), ("SRS + PPI++", "SRS + difference", "o", "-"),
           ("PPS-pop + Hájek (EPA-like)", "Population PPS", "s", "-"), ("SRS + GREG-ML", "ML-assisted GREG", "v", (0, (3, 1.5))),
           ("Neyman-H10 + HT (classical)", "Neyman allocation", "^", "-"), ("Neyman-H10 + PPI++ (hybrid)", "Neyman + difference", "^", (0, (3, 1.5))),
           ("ANCHOR (Poisson + PPI++)", "ANCHOR", "D", "-")]
NAME = {m: lab for m, lab, _, _ in METHODS}


def method_handles():
    return [Line2D([], [], color=C[m], lw=2.0 if m.startswith("ANCHOR") else 1.2, ls=ls, marker=mk, ms=4, mec=INK, mew=.45, label=lab)
            for m, lab, mk, ls in METHODS]


def eff_panel(pop, est, logy=False, ylim=None, yticks=None):
    def draw(fig, ss, ctx):
        if ctx["standalone"]:            # the combined figure shares one legend strip; a standalone panel carries its own
            gs = ss.subgridspec(2, 1, height_ratios=[.55, 1], hspace=.04)
            legend_strip(fig, gs[0], method_handles(), 2, labelspacing=.25, columnspacing=.8, handlelength=2.0); ss = gs[1]
        ax = fig.add_subplot(ss)
        ax.axhline(1, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
        for m, lab, mk, ls in METHODS:
            d = G[(G.population == pop) & (G.estimand == est) & (G.label == m)].sort_values("n")
            if not len(d):
                continue
            lw = 2.0 if m.startswith("ANCHOR") else 1.2
            ax.plot(d.n, d.rel_eff, color=C[m], lw=lw, ls=ls, marker=mk, ms=4, mec=INK, mew=.45, zorder=4 if m.startswith("ANCHOR") else 3)
        ax.set_xscale("log"); ax.set_xticks([100, 300, 1000], ["100", "300", "1,000"]); ax.minorticks_off(); ax.set_xlim(85, 1180)
        if logy:
            ax.set_yscale("log"); ax.minorticks_off()
        if ylim: ax.set_ylim(*ylim)
        if yticks: ax.set_yticks(yticks, [f"{t:g}" for t in yticks])
        ax.text(.03, .97, f"{pop.capitalize()} · {est}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        style_axis(ax, xlabel="Systems sampled", ylabel="Efficiency vs SRS" if (pop, est) in [("medium", "systems"), ("medium", "people")] else None)
    return draw


STRIP = .45


def panel_pareto(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.10)
    handles = [Line2D([], [], color=C["ANCHOR (Poisson + PPI++)"], lw=1.4, label="Dual"),
               Line2D([], [], ls="", marker="s", color=C["PPS-pop + Hájek (EPA-like)"], mec=INK, ms=5, label="Pop. PPS"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=4.5, label="Large"),
               Line2D([], [], ls="", marker="^", mfc="white", mec=INK, ms=4.5, label="Medium")]
    legend_strip(fig, gs[0], handles, 2, labelspacing=.25, columnspacing=.8, handlelength=1.4)
    ax = fig.add_subplot(gs[1])
    for pop, mk in [("large", "o"), ("medium", "^")]:
        d = G[(G.population == pop) & (G.n == 300) & G.label.str.startswith("dual-rho")]
        pv = d.pivot_table(index="label", values="rel_eff", columns="estimand").sort_index()
        ax.plot(pv.systems, pv.people, color=C["ANCHOR (Poisson + PPI++)"], lw=1.4, marker=mk, ms=4.5, mec=INK, mew=.45, zorder=3)
        e = G[(G.population == pop) & (G.n == 300) & (G.label == "PPS-pop + Hájek (EPA-like)")].set_index("estimand").rel_eff
        ax.plot(e["systems"], e["people"], ls="", marker=mk, ms=5.5, color=C["PPS-pop + Hájek (EPA-like)"], mec=INK, mew=.45, zorder=4)
    ax.plot(1, 1, ls="", marker="*", ms=8, color=C["SRS + Hájek (classical)"], mec=INK, mew=.45, zorder=4)
    ax.set_yscale("log"); ax.minorticks_off(); ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"]); ax.set_ylim(.7, 30)
    ax.set_xlim(.3, 2.1); ax.set_xticks([.5, 1, 1.5, 2])
    ax.annotate("SRS", (1, 1), xytext=(6, -2), textcoords="offset points", fontsize=8.5, va="top")
    style_axis(ax, xlabel="Efficiency, systems", ylabel="Efficiency, people")


def panel_coverage(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.10)
    legend_strip(fig, gs[0], [Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=4.5, label="Wald"),
                              Line2D([], [], ls="", marker="o", mfc=MUTED, mec=INK, ms=4.5, label="KG")], 2)
    ax = fig.add_subplot(gs[1])
    order = [m for m, _, _, _ in METHODS][::-1]
    y = np.arange(len(order))
    ax.axvline(.95, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    for i, m in enumerate(order):
        d = G[G.label == m]
        w, k = d.cov_wald.min(), d.cov_kg.min()
        ax.plot([w, k], [i, i], color=MUTED, lw=.8, zorder=2)
        ax.plot(w, i, ls="", marker="o", ms=4.5, mfc="white", mec=C[m] if m != "SRS + GREG-ML" else INK, mew=1.1, zorder=3)
        ax.plot(k, i, ls="", marker="o", ms=4.5, color=C[m], mec=INK, mew=.45, zorder=4)
    ax.set_yticks(y, [NAME[m] for m in order]); ax.set_xlim(.75, 1.0); ax.set_xticks([.8, .9, 1.0]); ax.set_ylim(-.6, len(order) - .4)
    style_axis(ax, xlabel="Minimum coverage")


ABL = [("ANCHOR (Poisson + PPI++)", "ANCHOR"), ("ANCHOR, tau = 0.25", "τ = 0.25"), ("ANCHOR, tau = 0.75", "τ = 0.75"),
       ("ANCHOR, lambda = 1", "λ fixed at 1"), ("ANCHOR systematic (fixed n)", "Fixed-size sample"), ("ANCHOR, sd rule", "√(f(1−f)) allocation"),
       ("ANCHOR design + GREG-lin", "GREG-lin estimator"), ("ANCHOR design + Hájek", "Hájek estimator"), ("SRS + PPI++", "Uniform allocation")]


def panel_ablation(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.04)
    legend_strip(fig, gs[0], [Patch(facecolor=C["systems"], edgecolor=INK, lw=.45, label="Systems"),
                              Patch(facecolor=C["people"], edgecolor=INK, lw=.45, label="People")], 2, handlelength=1.0)
    ax = fig.add_subplot(gs[1]); y = np.arange(len(ABL)); h = .38
    for off, est, key in [(-h / 2, "systems", "systems"), (h / 2, "people", "people")]:
        v = [G[(G.population == "medium") & (G.n == 300) & (G.estimand == est) & (G.label == m)].rel_eff.iloc[0] for m, _ in ABL]
        ax.barh(y + off, v, height=h, color=C[key], edgecolor=INK, lw=.45, label="Share of " + est, zorder=2)
    ax.axvline(1, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, [lab for _, lab in ABL]); ax.set_ylim(len(ABL) - .45, -.55); ax.set_xlim(0, 2.6); ax.set_xticks([0, 1, 2])
    style_axis(ax, xlabel="Efficiency vs SRS")


def layout(fig):
    g = fig.add_gridspec(3, 1, height_ratios=[.075, 1, 1.18], left=.06, right=.99, top=.99, bottom=.06, hspace=.1)
    strip = fig.add_subplot(g[0]); strip.axis("off")
    strip.legend(handles=method_handles(), loc="center", ncol=4, handletextpad=.35, columnspacing=1.0, handlelength=2.2)
    top = g[1].subgridspec(1, 4, wspace=.12); low = g[2].subgridspec(1, 3, width_ratios=[1, 1.25, 1.25], wspace=.12)
    return [top[0], top[1], top[2], top[3], low[0], low[1], low[2]]


if __name__ == "__main__":
    builders = [eff_panel("medium", "systems", ylim=(0, 2.7), yticks=[0, 1, 2]), eff_panel("large", "systems", ylim=(0, 2.4), yticks=[0, 1, 2]),
                eff_panel("medium", "people", ylim=(0, 3.2), yticks=[0, 1, 2, 3]), eff_panel("large", "people", logy=True, ylim=(.3, 90), yticks=[.3, 1, 3, 10, 30]),
                panel_pareto, panel_coverage, panel_ablation]
    save_modules(builders, "Fig4_census_validation", "Fig4_census_validation", layout, figsize_mm=(180, 150),
                 sizes_mm=[(85, 88), (85, 88), (85, 88), (85, 88), (62, 72), (75, 72), (75, 75)])
