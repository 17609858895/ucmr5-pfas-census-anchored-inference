"""Fig. 1 — the unmonitored majority: who UCMR5 measured, and what the small-system sample represents."""
from f_common import *

P = pd.read_csv(FD / "fig1_map_points.csv"); COV = pd.read_csv(FD / "fig1_coverage_by_class.csv")
COMP = pd.read_csv(FD / "fig1_small_composition.csv", index_col=0)
# drawing order: census first, the unmonitored majority on top of it, sampled small systems last
CATS = [("Census (≥3,300)", "census", 1.3, None, 0), ("Unmonitored", "unmonitored", .9, None, 0), ("Sampled small (<3,300)", "sampled", 5.5, INK, .25)]
LEGEND = {"Unmonitored": "Unmonitored", "Census (≥3,300)": "Larger (>3,300 people)", "Sampled small (<3,300)": "Sampled (≤3,300 people)"}


def panel_a(fig, ss, ctx):
    ax = fig.add_subplot(ss); basemap(ax, lw=.25)
    for name, key, s, ec, lw in CATS:
        d = P[P.category == name].dropna(subset=["lat"])
        map_points(ax, d.lon.to_numpy(), d.lat.to_numpy(), s=s, color=C[key], edgecolor=ec if ec else "none", linewidths=lw, zorder=3)
    n = P.category.value_counts()
    handles = [Line2D([], [], ls="", marker="o", mfc=C[k], mec=INK if ec else C[k], mew=.4, ms=4.2, label=f"{LEGEND[nm]} (n = {n[nm]:,})")
               for nm, k, _, ec, _ in [CATS[1], CATS[0], CATS[2]]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .0), ncol=3, handletextpad=.3, columnspacing=1.2)


def panel_b(fig, ss, ctx):
    ax = fig.add_subplot(ss); x = np.arange(len(COV)); w = .38
    for off, col, key, lab in [(-w / 2, "share_systems", "systems", "Systems"), (w / 2, "share_people", "people", "People served")]:
        ax.bar(x + off, COV[col] * 100, width=w, color=C[key], edgecolor=INK, lw=.5, label=lab, zorder=2)
    ax.set_yscale("log"); ax.set_ylim(.2, 150); ax.set_yticks([.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"]); ax.minorticks_off()
    ax.set_xticks(x, COV.cls, rotation=30, ha="right")
    ax.axvline(3.5, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.legend(loc="upper left", handlelength=1.1)
    style_axis(ax, xlabel="People served by system", ylabel="Monitored in UCMR5 (%)")


def panel_c(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[.5, 1], hspace=.06)
    handles = [Patch(facecolor=C[f"cls{i}"], edgecolor=INK, lw=.5, label=c) for i, c in enumerate(CLASS_ORDER)]
    legend_strip(fig, gs[0], handles, 2, title="People served by system", handlelength=1.0)
    ax = fig.add_subplot(gs[1]); rows = ["Frame: systems", "Frame: people", "UCMR5 sample: systems"]
    left = np.zeros(len(rows))
    for i, cls in enumerate(CLASS_ORDER):
        v = COMP.loc[cls, rows].to_numpy() * 100
        ax.barh(np.arange(len(rows)), v, left=left, color=C[f"cls{i}"], edgecolor=INK, lw=.5, height=.62, label=cls, zorder=2)
        for j, (l, vv) in enumerate(zip(left, v)):
            if vv >= 10:
                ax.text(l + vv / 2, j, f"{vv:.0f}", ha="center", va="center", fontsize=8.5, zorder=3)
        left += v
    ax.set_yticks(np.arange(len(rows)), ["Systems", "People", "UCMR5 sample"]); ax.set_ylim(2.55, -.55)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    style_axis(ax, xlabel="Share within small systems (%)")


def layout(fig):
    g = fig.add_gridspec(2, 1, height_ratios=[1.12, 1], left=.06, right=.985, top=.985, bottom=.06, hspace=.10)
    low = g[1].subgridspec(1, 2, width_ratios=[1, 1.08], wspace=.32)
    return [g[0], low[0], low[1]]


if __name__ == "__main__":
    save_modules([panel_a, panel_b, panel_c], "Fig1_monitoring_gap", "Fig1_monitoring_gap", layout,
                 figsize_mm=(180, 150), sizes_mm=[(170, 105), (90, 75), (95, 72)])
