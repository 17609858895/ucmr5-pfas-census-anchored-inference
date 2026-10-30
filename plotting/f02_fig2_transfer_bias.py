"""Fig. 2 — census-trained models rank small systems well but under-predict their risk."""
from f_common import *

ROC = pd.read_csv(FD / "fig2_roc.csv"); CURVE = pd.read_csv(FD / "fig2_size_curve.csv"); CLS = pd.read_csv(FD / "fig2_by_class.csv")
MODELS = [("SDWIS attributes", "m_sdwis", "SDWIS"), ("+ geography", "m_geo", "+ Geography"), ("+ census neighbour labels", "m_nb", "+ Neighbours")]
STRIP = .36   # legend strip height relative to the axes
POP_TICKS = [2, 3, 4, 5, 6]; POP_LABELS = ["10²", "10³", "10⁴", "10⁵", "10⁶"]


def panel_a(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.05)
    handles = [Line2D([], [], color=C[key], lw=1.6, label=f"{short} ({ROC[ROC.model == name].auc.iloc[0]:.2f})") for name, key, short in MODELS]
    legend_strip(fig, gs[0], handles, 1, handlelength=1.2, labelspacing=.25)
    ax = fig.add_subplot(gs[1])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    for name, key, _ in MODELS:
        d = ROC[ROC.model == name]
        ax.plot(d.fpr, d.tpr, color=C[key], lw=1.6, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    style_axis(ax, xlabel="False-positive rate", ylabel="True-positive rate")


def panel_b(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.05)
    handles = [Line2D([], [], ls="", marker="o", mfc=C["observed"], mec=INK, mew=.5, ms=5, label="Design estimate"),
               Line2D([], [], ls="", marker="D", mfc=C["plugin"], mec=INK, mew=.5, ms=4.5, label="Predicted")]
    legend_strip(fig, gs[0], handles, 1, labelspacing=.25)
    ax = fig.add_subplot(gs[1]); x = np.arange(len(CLS))
    ax.errorbar(x - .12, CLS.obs * 100, yerr=[(CLS.obs - CLS.lo) * 100, (CLS.hi - CLS.obs) * 100], fmt="o", ms=5,
                mfc=C["observed"], mec=INK, mew=.5, ecolor=INK, elinewidth=.8, capsize=2, label="Observed (UCMR5 sample)", zorder=3)
    ax.scatter(x + .12, CLS.plugin * 100, marker="D", s=26, color=C["plugin"], edgecolor=INK, lw=.5, label="Census model prediction", zorder=4)
    ax.set_xticks(x, CLS.cls, rotation=30, ha="right"); ax.set_xlim(-.5, len(CLS) - .5)
    ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20])
    style_axis(ax, xlabel="People served by system", ylabel="Systems above MCL (%)")


def panel_c(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[STRIP, 1], hspace=.05)
    eb = dict(ls="", marker="o", mec=INK, mew=.5, ms=4.8)
    handles = [Line2D([], [], mfc=C["census"], label="Census obs.", **eb), Line2D([], [], color=C["census"], lw=1.6, label="Census pred."),
               Line2D([], [], mfc=C["observed"], label="Small obs.", **eb), Line2D([], [], color=C["plugin"], lw=1.6, ls=(0, (4, 2)), label="Small pred.")]
    legend_strip(fig, gs[0], handles, 2, handlelength=1.5, labelspacing=.25, columnspacing=.7)
    ax = fig.add_subplot(gs[1])
    ax.axvline(np.log10(3300), color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
    cen, sm = CURVE[CURVE.group == "census"], CURVE[CURVE.group == "small"]
    ax.errorbar(cen.lpop, cen.obs * 100, yerr=[(cen.obs - cen.lo) * 100, (cen.hi - cen.obs) * 100], fmt="o", ms=4.5, mfc=C["census"], mec=INK,
                mew=.5, ecolor=INK, elinewidth=.8, capsize=2, label="Census: observed", zorder=3)
    ax.plot(cen.lpop, cen.pred * 100, color=C["census"], lw=1.6, label="Census: predicted (out-of-fold)", zorder=2)
    ax.plot(sm.lpop, sm.pred * 100, color=C["plugin"], lw=1.6, ls=(0, (4, 2)), label="Small: predicted", zorder=2)
    ax.errorbar(CLS.lpop, CLS.obs * 100, yerr=[(CLS.obs - CLS.lo) * 100, (CLS.hi - CLS.obs) * 100], fmt="o", ms=5, mfc=C["observed"], mec=INK,
                mew=.5, ecolor=INK, elinewidth=.8, capsize=2, label="Small: observed", zorder=4)
    ax.set_xlim(1.4, 6.6); ax.set_xticks(POP_TICKS, POP_LABELS); ax.set_ylim(0, 40); ax.set_yticks([0, 10, 20, 30, 40])
    style_axis(ax, xlabel="People served by system", ylabel="Systems above MCL (%)")


def layout(fig):
    g = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=.06, right=.99, top=.98, bottom=.1, wspace=.30)
    return [g[0], g[1], g[2]]


if __name__ == "__main__":
    save_modules([panel_a, panel_b, panel_c], "Fig2_transfer_bias", "Fig2_transfer_bias", layout,
                 figsize_mm=(180, 84), sizes_mm=[(62, 80), (62, 80), (78, 80)])
