"""Fig. 6 — independent state data: state-level prevalence transfers after recalibration; within-state ranking weakens."""
from f_common import *

FU = pd.read_csv(FD / "fig6_external_validation_followup.csv").set_index("item")
AUC = pd.read_csv(FD / "fig6_auc.csv"); REC = pd.read_csv(FD / "fig6_recall_curves.csv")
ESTS = [("plug-in mean(f)", "ML prediction", "s", "plugin"), ("ratio estimator k*mean(f)", "Ratio", "o", "ratio"),
        ("logit-shift estimator", "Logit shift", "D", "recal"), ("additive estimator mean(f)+rectifier", "Additive", "^", "additive")]
STATES = [("MI", "Michigan"), ("MN", "Minnesota")]
SETS = ["UCMR5 small sample (national)", "Minnesota", "New Hampshire", "Michigan"]
SHORT = {"UCMR5 small sample (national)": "UCMR5 sample", "Michigan": "Michigan", "Minnesota": "Minnesota", "New Hampshire": "New Hampshire"}
STRIP = .3


def prev_handles():
    h = [Line2D([], [], ls="", marker=mk, ms=5, color=C[col], mec=INK, mew=.5, label=lab) for _, lab, mk, col in ESTS]
    return h + [Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label="State data")]


def panel_prev(fig, ss, ctx):
    if ctx["standalone"]:
        gs = ss.subgridspec(2, 1, height_ratios=[.32, 1], hspace=.04)
        legend_strip(fig, gs[0], prev_handles(), 3, labelspacing=.25, columnspacing=.8, handlelength=1.4); ss = gs[1]
    ax = fig.add_subplot(ss); ticks, labels = [], []
    for gi, (st, name) in enumerate(STATES):
        base = gi * 5.2
        truth = FU.loc[f"{st}: state truth (tested small systems)", "value"]
        ax.plot([truth * 100] * 2, [base - .6, base + len(ESTS) - .4], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)
        for ei, (key, lab, mk, col) in enumerate(ESTS):
            r = FU.loc[f"{st}: {key}"]; y = base + ei
            if np.isfinite(r.lo):
                ax.plot([r.lo * 100, r.hi * 100], [y, y], color=INK, lw=.9, zorder=2)
            ax.plot(r.value * 100, y, ls="", marker=mk, ms=5, color=C[col], mec=INK, mew=.5, zorder=3)
        ticks.append(base + 1.5); labels.append(name)
    ax.set_yticks(ticks, labels); ax.set_ylim(5.2 + len(ESTS) - .3, -.7)
    ax.set_xlim(0, 8); ax.set_xticks([0, 2, 4, 6, 8])
    style_axis(ax, xlabel="Small systems above MCL (%)")


def panel_auc(fig, ss, ctx):
    ax = fig.add_subplot(ss); y = np.arange(len(SETS))
    ax.axvline(.5, color=MUTED, lw=.9, ls=(0, (3, 2)), zorder=1); ax.axvline(.7, color=MUTED, lw=.9, ls=(0, (1, 1.5)), zorder=1)
    for i, s in enumerate(SETS):
        r = AUC[AUC.dataset == s].iloc[0]
        ax.plot([r.lo, r.hi], [i, i], color=INK, lw=.9, zorder=2)
        ax.plot(r.auc, i, ls="", marker="o", ms=5.5, color=C[s], mec=INK, mew=.5, zorder=3)
    ax.set_yticks(y, [SHORT[s] + "\n" + f"(n = {AUC[AUC.dataset == s].n.iloc[0]:,})" for s in SETS]); ax.set_ylim(len(SETS) - .45, -.55)
    ax.set_xlim(.4, 1.0); ax.set_xticks([.4, .6, .8, 1.0])
    style_axis(ax, xlabel="AUC")


def panel_recall(fig, ss, ctx):
    ax = fig.add_subplot(ss)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    for s in SETS:
        d = REC[REC.set == s]
        ax.plot(d.tested, d.found, color=C[s], lw=1.6, zorder=3 if s != "Michigan" else 4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ABBR = {"UCMR5 small sample (national)": "UCMR5", "Michigan": "MI", "Minnesota": "MN", "New Hampshire": "NH"}
    ax.legend(handles=[Line2D([], [], color=C[s], lw=1.6, label=ABBR[s]) for s in SETS], loc="lower right", ncol=1, handlelength=1.1,
              labelspacing=.2, borderaxespad=.1)
    style_axis(ax, xlabel="Systems tested (risk-ranked)", ylabel="Exceedances found")


def layout(fig):
    g = fig.add_gridspec(2, 1, height_ratios=[.08, 1], left=.06, right=.99, top=.99, bottom=.1, hspace=.06)
    strip = fig.add_subplot(g[0]); strip.axis("off")
    strip.legend(handles=prev_handles(), loc="center", ncol=5, handletextpad=.3, columnspacing=1.2, handlelength=1.5)
    row = g[1].subgridspec(1, 3, width_ratios=[1, 1.1, .95], wspace=.12)
    return [row[0], row[1], row[2]]


if __name__ == "__main__":
    save_modules([panel_prev, panel_auc, panel_recall], "Fig6_external_validation", "Fig6_external_validation", layout,
                 figsize_mm=(180, 74), sizes_mm=[(80, 80), (70, 66), (62, 66)])

