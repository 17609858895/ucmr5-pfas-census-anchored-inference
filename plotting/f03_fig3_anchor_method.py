"""Fig. 3 — ANCHOR: census-anchored design and two-phase prediction-powered estimation (vector schematic + allocation)."""
from matplotlib.patches import FancyBboxPatch
from f_common import *

PI = pd.read_csv(FD / "fig3_inclusion_probabilities.csv")
W, H = 124.0, 80.0          # drawing units = millimetres of the schematic cell
FS = 8.5


def _box(ax, x, y, w, h, fc, title, lines=(), r=1.4, t_off=3.4, l_off=8.4, dy=4.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=MUTED, lw=.7, zorder=1))
    ax.text(x + w / 2, y + h - t_off, title, ha="center", va="center", fontsize=FS, zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - l_off - dy * i, t, ha="center", va="center", fontsize=FS, zorder=3)


def _arrow(ax, p0, p1, label=None, dx=0, dy=1.6):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>,head_length=.45,head_width=.22", color=INK, lw=.8, shrinkA=0, shrinkB=0), zorder=4)
    if label:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=FS, zorder=4)


def panel_schematic(fig, ss, ctx):
    import json
    ax=fig.add_subplot(ss); ax.set_gid("schematic"); ax.axis("off")
    src=OUTPUT/"figures/Fig3_anchor_method/source"
    ax.imshow(plt.imread(src/"schematic_background.png"),extent=[0,1,1,0],aspect="auto")
    for x,y,t in json.loads((src/"label_positions.json").read_text(encoding="utf-8")):
        ax.text(x,y,t.replace("\\n","\n"),ha="center",va="center",fontsize=10 if y==.047 else 8.5,color=INK)

def panel_equation(fig,ss,ctx):
    ax=fig.add_subplot(ss); ax.set_gid("schematic");ax.axis("off")
    ax.text(.02,.93,"Two-phase difference estimator",fontsize=9.5,va="top")
    lines=[r"$\hat\theta=\frac{1}{A}[\lambda\sum_U a_i f_i$",r"$+\sum_s\frac{a_i}{\pi_i}(\tilde y_i-\lambda f_i)$",r"$+\sum_{s_2}\frac{a_i}{\pi_i q}(y_i-\tilde y_i)]$"]
    for i,txt in enumerate(lines):ax.text(.04,.74-.23*i,txt,fontsize=11,va="center")
    ax.text(.02,.08,"a: 1 or people served; A: frame total\nỹ: reduced schedule; y: full schedule",fontsize=8.5,va="center")


def panel_alloc(fig, ss, ctx):
    gs = ss.subgridspec(2, 1, height_ratios=[.3, 1], hspace=.04)
    designs = [("pi_epa", "Reconstructed cells", C["PPS-pop + Hájek (EPA-like)"], "s"), ("pi_anchor", "ANCHOR (systems)", C["ANCHOR (Poisson + PPI++)"], "D"),
               ("pi_dual", "Dual design", C["Neyman-H10 + PPI++ (hybrid)"], "o")]
    legend_strip(fig, gs[0], [Line2D([], [], color=c, lw=1.4, marker=m, ms=4, mec=INK, mew=.45, label=l) for _, l, c, m in designs], 1, labelspacing=.25)
    ax = fig.add_subplot(gs[1])
    bins = np.quantile(np.log10(PI.f), np.linspace(0, 1, 11)); b = np.clip(np.digitize(np.log10(PI.f), bins[1:-1]), 0, 9)
    x = PI.groupby(b).f.median().values
    for col, lab, c, m in designs:
        med = PI.groupby(b)[col].median().values; lo = PI.groupby(b)[col].quantile(.1).values; hi = PI.groupby(b)[col].quantile(.9).values
        ax.fill_between(x, lo * 100, hi * 100, color=c, alpha=.2, lw=0, zorder=1)
        ax.plot(x, med * 100, color=c, lw=1.4, marker=m, ms=4, mec=INK, mew=.45, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks([.003, .01, .03, .1, .3], ["0.3", "1", "3", "10", "30"]); ax.set_yticks([.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax.set_ylim(.2, 15)
    style_axis(ax, xlabel="Predicted risk (%)", ylabel="Inclusion probability (%)")


def layout(fig):
    g=fig.add_gridspec(2,1,height_ratios=[1.85,1],left=.035,right=.985,top=.99,bottom=.065,hspace=.09)
    low=g[1].subgridspec(1,2,width_ratios=[1.03,1],wspace=.12)
    return [g[0],low[0],low[1]]

if __name__=="__main__":
    # Use the current three-row figure; the formula belongs in Methods 2.6.
    import generate_refined_figures as current
    builders,grid,size,panels,folder=current.TASKS['3']
    current.save('3',builders,grid,size,panels,folder)
