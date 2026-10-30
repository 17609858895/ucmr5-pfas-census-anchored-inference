"""Fig. 5 — PFOA/PFOS burden in small systems: national, by size, and by state."""
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from f_common import *

B = pd.read_csv(FD / "fig5_burden.csv"); S = pd.read_csv(FD / "fig5_state_ratio_estimates.csv")
# design-only uses the colour of the classical estimator in Fig. 4
EST = [("design-only (post-stratified Hajek)", "Design only", "o", "SRS + Hájek (classical)"), ("ANCHOR PPI++", "ANCHOR", "D", "ANCHOR (Poisson + PPI++)"),
       ("ML plug-in (census model, no rectification)", "ML prediction", "s", "plugin")]
SIZE_KEY = {"≤100": "size <=100", "101–500": "size 101-500", "501–1,000": "size 501-1000", "1,001–3,300": "size 1001-3300"}
# continuous scale between palette colours (white -> PAL[0] -> PAL[4]); square-root stretch for the many low-prevalence states
CMAP = LinearSegmentedColormap.from_list("white_rose_lavender", ["#FFFFFF", PAL[0], PAL[4]])
NORM = PowerNorm(.5, 0, .5)


def panel_map(fig, ss, ctx):
    ax = fig.add_subplot(ss)
    basemap(ax, values=dict(zip(S.ST, S.prev)), cmap=CMAP, norm=NORM, lw=.3)
    hcolorbar(ax, CMAP, NORM, "Small systems above MCL (%)", ticks=[0, .02, .1, .25, .5], fmt=lambda t: f"{t * 100:.0f}",
              pos=(.62, .07, .33, .045))


def panel_states(fig, ss, ctx):
    ax = fig.add_subplot(ss)
    d = S[S.ST.str.len() == 2].sort_values("people_exc", ascending=False).head(12)
    y = np.arange(len(d))
    ax.barh(y, d.people_exc / 1e3, height=.65, color=C["ANCHOR (Poisson + PPI++)"], edgecolor=INK, lw=.45, zorder=2)
    ax.errorbar(d.people_exc / 1e3, y, xerr=[(d.people_exc - d.people_exc_lo) / 1e3, (d.people_exc_hi - d.people_exc) / 1e3],
                fmt="none", ecolor=INK, elinewidth=.8, capsize=1.8, zorder=3)
    ax.set_yticks(y, d.ST); ax.set_ylim(len(d) - .4, -.6); ax.set_xlim(0, 310); ax.set_xticks([0, 100, 200, 300])
    style_axis(ax, xlabel="People above MCL (thousands)")


def est_handles(keys=("Design only", "ANCHOR", "ML prediction")):
    return [Line2D([], [], ls="", marker=mk, ms=5, color=C[col], mec=INK, mew=.5, label=lab) for _, lab, mk, col in EST if lab in keys]


def own_legend(fig, ss, ctx, keys):
    """The combined figure shares one legend strip; a standalone panel carries its own."""
    if not ctx["standalone"]:
        return ss
    gs = ss.subgridspec(2, 1, height_ratios=[.14, 1], hspace=.04)
    legend_strip(fig, gs[0], est_handles(keys), len(keys))
    return gs[1]


def panel_national(fig, ss, ctx):
    ss = own_legend(fig, ss, ctx, ("Design only", "ANCHOR", "ML prediction"))
    ax = fig.add_subplot(ss); rows = []
    for gi, est in enumerate(["systems", "people"]):
        for ei, (key, lab, mk, col) in enumerate(EST):
            r = B[(B.group == "all") & (B.estimand == est) & (B.estimator == key)].iloc[0]
            y = gi * 4 + ei
            if np.isfinite(r.kg_lo):
                ax.plot([r.kg_lo * 100, r.kg_hi * 100], [y, y], color=INK, lw=.9, zorder=2)
            ax.plot(r.theta * 100, y, ls="", marker=mk, ms=5, color=C[col], mec=INK, mew=.5, zorder=3)
            rows.append((y, lab))
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows]); ax.set_ylim(6.6, -1.5)
    for gi, est in enumerate(["Share of systems", "Share of people"]):
        ax.text(.25, gi * 4 - .95, est, ha="left", va="center", fontsize=8.5)
    ax.set_xlim(0, 13); ax.set_xticks([0, 4, 8, 12])
    style_axis(ax, xlabel="Above MCL (%)")


def panel_size(fig, ss, ctx):
    ss = own_legend(fig, ss, ctx, ("ANCHOR", "ML prediction"))
    ax = fig.add_subplot(ss); x = np.arange(len(SIZE_KEY))
    for (cls, key), xi in zip(SIZE_KEY.items(), x):
        a = B[(B.group == key) & (B.estimand == "systems") & (B.estimator == "ANCHOR PPI++")].iloc[0]
        p = B[(B.group == key) & (B.estimand == "systems") & (B.estimator.str.startswith("ML plug-in"))].iloc[0]
        ax.errorbar(xi - .12, a.theta * 100, yerr=[[(a.theta - a.kg_lo) * 100], [(a.kg_hi - a.theta) * 100]], fmt="D", ms=4.8,
                    mfc=C["ANCHOR (Poisson + PPI++)"], mec=INK, mew=.5, ecolor=INK, elinewidth=.8, capsize=2, zorder=3)
        ax.plot(xi + .12, p.theta * 100, ls="", marker="s", ms=4.6, color=C["plugin"], mec=INK, mew=.5, zorder=3)
    ax.set_xticks(x, list(SIZE_KEY), rotation=30, ha="right"); ax.set_xlim(-.5, len(x) - .5)
    ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20])
    style_axis(ax, xlabel="People served by system", ylabel="Systems above MCL (%)")


def layout(fig):
    g = fig.add_gridspec(3, 1, height_ratios=[1.25, .07, 1], left=.06, right=.99, top=.99, bottom=.07, hspace=.08)
    top = g[0].subgridspec(1, 2, width_ratios=[2.1, 1], wspace=.06)
    strip = fig.add_subplot(g[1]); strip.axis("off")
    strip.legend(handles=est_handles(),
                 loc="center", ncol=3, handletextpad=.3, columnspacing=1.4)
    low = g[2].subgridspec(1, 2, width_ratios=[1.15, 1], wspace=.14)
    return [top[0], top[1], low[0], low[1]]


if __name__ == "__main__":
    save_modules([panel_map, panel_states, panel_national, panel_size], "Fig5_small_burden", "Fig5_small_burden", layout,
                 figsize_mm=(180, 140), sizes_mm=[(130, 85), (60, 80), (90, 68), (80, 68)])
