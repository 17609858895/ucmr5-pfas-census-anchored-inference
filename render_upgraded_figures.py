"""Redesigned Fig. 5 (burden), Fig. 7 (external monitoring designs) and the graphical abstract.

Usage:  python render_upgraded_figures.py --workdir <rendering workdir prepared by rebuild_figures.py>

Inputs are frozen tables only (no fitting, bootstrapping or re-estimation):
  figure_data/fig5_burden.csv, fig5c_additive_domain_shares.csv, fig5c_full_inventory_shares.csv,
  weight_calibration_sensitivity.csv, timing_burden_sensitivity.csv, within_cell_selection_odds_curve.csv
  results/strengthening/method/external_primary_R12_B300.csv
  (graphical-abstract workflow drawn by workflow_schematic.py; no raster artwork)
Outputs: <workdir>/figures/Fig_5, Fig_7 (combined PNG/PDF + independently redrawn subfigures) and Graphical_Abstract.png/pdf.
"""
from pathlib import Path
import argparse, os, sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy.stats import binomtest

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--workdir', type=Path, default=Path(os.environ.get('ANCHOR_WORKDIR', '.')))
ap.add_argument('--only', nargs='+', choices=['5', '7', 'GA'], default=['5', '7', 'GA'])
ARGS = ap.parse_args()
WORK = ARGS.workdir.resolve()
HELPERS = Path(__file__).resolve().parent / 'strengthening/method/figure_helpers/figkit'
sys.path.insert(0, str(HELPERS)); sys.path.insert(0, str(HELPERS / 'publication_helpers'))
import figstyle as fs
import publication_layout as layout
from ml_publication_style import configure_publication_style, style_axis
from scientific_palettes import get_palette
from audit_palette_usage import validate_series_colors
from matplotlib import pyplot as plt

FIG = WORK / 'figures'; QA = WORK / 'figure_QA_upgraded'; QA.mkdir(parents=True, exist_ok=True)
fs.OUTPUT = WORK; fs.FIG = FIG; fs.QA = QA; fs.MIN_PT = 12; fs.LETTER_PT = 12
layout.OUTPUT = WORK; layout.FIG = FIG
configure_publication_style()
plt.rcParams.update({'font.size': 12, 'font.weight': 'normal', 'axes.labelsize': 13.5, 'axes.labelweight': 'normal',
                     'axes.titleweight': 'normal', 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12,
                     'axes.linewidth': 1., 'lines.linewidth': 1.4, 'savefig.dpi': 600, 'hatch.linewidth': .8})
PAL = get_palette('nature_new'); INK = fs.INK; MUTED = fs.MUTED
# one series -> one palette colour, identical to the legacy map used by Fig. 1-6
COL = {'systems': PAL[4], 'people': PAL[3], 'post': PAL[6], 'assisted': PAL[3], 'model_only': PAL[1], 'highlight': PAL[6]}
validate_series_colors('nature_new', COL)
FD = WORK / 'figure_data'


def sty(ax, x=None, y=None):
    style_axis(ax, xlabel=x, ylabel=y, tick_size=12.5, label_size=13.5)
    ax.tick_params(length=4.5, width=1, pad=3)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.)
    return ax


def legend_strip(fig, ss, handles, ncol, **kw):
    ax = fig.add_subplot(ss); ax.axis('off')
    opts = {'handlelength': 1.2, 'handletextpad': .4, 'columnspacing': 1.3, 'borderpad': 0}; opts.update(kw)
    ax.legend(handles=handles, loc='center', ncol=ncol, **opts)
    return ax


# ----------------------------------------------------------------------------------------------- Fig. 5
B = pd.read_csv(FD / 'fig5_burden.csv')
SHARE = pd.read_csv(FD / 'fig5c_additive_domain_shares.csv')
INVENTORY = pd.read_csv(FD / 'fig5c_full_inventory_shares.csv')
SENS = pd.read_csv(FD / 'weight_calibration_sensitivity.csv')
TIMING = pd.read_csv(FD / 'timing_burden_sensitivity.csv')
ODDS = pd.read_csv(FD / 'within_cell_selection_odds_curve.csv')
EST = [('design-only (post-stratified Hajek)', 'Post-stratified', 'post'), ('ANCHOR PPI++', 'Model-assisted', 'assisted'),
       ('ML plug-in (census model, no rectification)', 'Model-only', 'model_only')]
CLASSES = ['<=100', '101-500', '501-1000', '1001-3300']
CLASS_LABELS = ['≤100', '101–500', '501–1,000', '1,001–3,300']


def national_counts(fig, ss, ctx):
    """Counts of exceeding systems and associated service population; the hatched segment is the burden that
    the uncorrected larger-system prediction omits relative to the model-assisted estimate."""
    row = ss.subgridspec(1, 2, wspace=.10)
    for j, (target, scale, unit, xmax, ticks) in enumerate([('systems', 1e3, 'Exceeding systems (thousands)', 8.6, [0, 2, 4, 6, 8]),
                                                             ('people', 1e6, 'People served (millions)', 3.3, [0, 1, 2, 3])]):
        ax = fig.add_subplot(row[j]); rows = {}
        for y, (key, label, k) in enumerate(EST):
            r = B[(B.group == 'all') & (B.estimand == target) & (B.estimator == key)].iloc[0]; rows[k] = r
            ax.barh(y, r['count'] / scale, height=.58, color=COL[k], edgecolor=INK, lw=.6, zorder=2)
            if np.isfinite(r.count_lo):
                ax.errorbar(r['count'] / scale, y, xerr=[[(r['count'] - r.count_lo) / scale], [(r.count_hi - r['count']) / scale]],
                            fmt='none', ecolor=INK, elinewidth=1, capsize=2.5, zorder=3)
        a, m = rows['assisted'], rows['model_only']
        ax.barh(2, (a['count'] - m['count']) / scale, left=m['count'] / scale, height=.58, color='white',
                edgecolor=COL['assisted'], hatch='////', lw=.8, zorder=1)
        gap = (a['count'] - m['count'])

        txt = f"+{gap:,.0f} missed" if target == 'systems' else f"+{gap / 1e6:.2f} M missed"
        ax.text(a['count'] / scale + .03 * xmax, 2, txt, ha='left', va='center', fontsize=12)
        for y, k in enumerate(['post', 'assisted']):
            r = rows[k]
            ax.text(r.count_hi / scale + .03 * xmax, y, f"{r.theta * 100:.2f}%", ha='left', va='center', fontsize=12)
        ax.set_yticks(range(3), [l for _, l, _ in EST] if j == 0 else [''] * 3)
        ax.set_ylim(2.55, -.55); ax.set_xlim(0, xmax); ax.set_xticks(ticks)
        sty(ax, unit)


def weight_sensitivity(fig, ss, ctx):
    ax = fig.add_subplot(ss); d = SENS.iloc[[0, 1, 2, 3, 6, 7]].copy()
    values = [*list(d.systems_difference * 100), TIMING.loc[TIMING.scheme == 'span_le366_days', 'systems_difference'].iloc[0] * 100]
    labels = ['Equal', 'Source', 'Size', 'Source × size', '+ Region', 'Exclude 3,300', 'Span ≤366 d']
    y = np.arange(len(values))
    primary = d.loc[d.scheme == 'Source-by-size (primary)', 'systems_difference'].iloc[0] * 100
    ax.axvline(primary, color=MUTED, lw=.8, ls='--', zorder=1)
    ax.hlines(y, primary, values, color=MUTED, lw=.9, zorder=2)
    ax.plot(values, y, ls='', marker='o', ms=6.2, color=COL['systems'], mec=INK, mew=.6, zorder=3)
    ax.plot(primary, 3, ls='', marker='o', ms=9.5, mfc='none', mec=INK, mew=.8, zorder=4)
    ax.set_yticks(y, labels); ax.set_ylim(len(values) - .45, -.6)
    ax.set_xlim(6.5, 8.3); ax.set_xticks([6.5, 7, 7.5, 8])
    sty(ax, 'Exceeding systems (%)')


def selection_odds(fig, ss, ctx):
    ax = fig.add_subplot(ss)
    ax.fill_between(ODDS.gamma, 0, ODDS.system_prevalence * 100, color=COL['systems'], alpha=.18, lw=0, zorder=1)
    ax.plot(ODDS.gamma, ODDS.system_prevalence * 100, color=COL['systems'], lw=1.8, zorder=3)
    r = ODDS.iloc[np.argmin(abs(ODDS.gamma - 1))]
    ax.plot(r.gamma, r.system_prevalence * 100, ls='', marker='o', ms=6.2, mfc=COL['systems'], mec=INK, mew=.6, zorder=4)
    ax.axvline(1, color=MUTED, lw=.8, ls='--', zorder=2)
    ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks([.25, .5, 1, 2, 4], ['0.25', '0.5', '1', '2', '4'])
    ax.set_xlim(.25, 4); ax.set_ylim(0, 27); ax.set_yticks([0, 10, 20])
    sty(ax, 'Exceedance-odds ratio\n(unmonitored / observed)', 'Exceeding systems (%)')


def butterfly(fig, ss, ctx):
    """Size-domain shares of exceeding systems (left) and of their service population (right)."""
    g = ss.subgridspec(2, 1, height_ratios=[.13, 1], hspace=.05)
    legend_strip(fig, g[0], [Patch(fc=COL['systems'], ec=INK, lw=.6, label='Exceeding systems'),
                             Patch(fc=COL['people'], ec=INK, lw=.6, label='Associated service population'),
                             Line2D([], [], ls='', marker='|', ms=13, mew=2.2, color=INK, label='Inventory share')], 3)
    ax = fig.add_subplot(g[1]); y = np.arange(4)
    for target, sign in [('systems', -1), ('people', 1)]:
        d = SHARE[SHARE.estimand == target].set_index('class').loc[CLASSES]
        inv = INVENTORY[INVENTORY.estimand == target].set_index('class').loc[CLASSES].inventory_share_pct
        ax.barh(y, sign * d.share_pct, height=.62, color=COL[target], edgecolor=INK, lw=.6, zorder=2)
        low, high = d.share_pct - d.lo_pct, d.hi_pct - d.share_pct
        ax.errorbar(sign * d.share_pct, y, xerr=[high, low] if sign < 0 else [low, high], fmt='none', ecolor=INK,
                    elinewidth=.9, capsize=2.2, zorder=3)
        ax.plot(sign * inv, y, ls='', marker='|', ms=19, mew=2.2, color=INK, zorder=4)
        for yi, v in zip(y, d.share_pct):
            ax.text(sign * (d.hi_pct.iloc[yi] + 2.5), yi, f"{v:.0f}%", ha='right' if sign < 0 else 'left', va='center', fontsize=12)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, CLASS_LABELS); ax.set_ylim(3.55, -.55)
    ax.set_xlim(-82, 82); ax.set_xticks([-75, -50, -25, 0, 25, 50, 75])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{abs(v):g}'))
    sty(ax, 'Share within the small-system burden (%)', 'System size (people)')


def layout5(fig):
    g = fig.add_gridspec(3, 1, height_ratios=[.92, 1.0, 1.02], left=.065, right=.99, top=.99, bottom=.045, hspace=.13)
    middle = g[1].subgridspec(1, 2, wspace=.13)
    return [g[0], middle[0], middle[1], g[2]]


# ----------------------------------------------------------------------------------------------- Fig. 7
PRIMARY = pd.read_csv(WORK / 'results/strengthening/method/external_primary_R12_B300.csv')
PRIMARY = PRIMARY[~PRIMARY.coefficient.eq('one')]
# same-depth classical control (external_neyman_two_phase.py), budget equal to 300 full schedules
NEY2 = pd.read_csv(WORK / 'results/strengthening/method/external_neyman_two_phase.csv')
NEY2 = NEY2[NEY2.budget_full_equivalents.eq(300)].assign(policy='neyman5_two_phase')
PRIMARY = pd.concat([PRIMARY, NEY2[['population', 'estimand', 'policy', 'mse_ratio_to_SRS_Hajek', 'coverage_kg', 'draws']]],
                    ignore_index=True)
DESIGNS = [('srs_full_hajek', 'SRS · full'), ('neyman5_full_difference', 'Neyman† · full'), ('pps_full_hajek', 'PPS · full'),
           ('dual_full', 'Dual · full'), ('neyman5_two_phase', 'Neyman† · 20%'), ('system_anchor_two_phase', 'System · 20%'),
           ('uniform20', 'Dual · 20%'), ('pps_two_phase_hajek', 'PPS · 20%'), ('uncorrected', 'Dual · R12 only')]
GROUPS = [3.5, 7.5]          # separators: full schedules | R12 + 20% completion | R12 without completion
MK = {'systems': 'o', 'people': 'D'}
STATES = {'MI': 'Michigan (n = 1,283)', 'WA': 'Washington (n = 855)'}


def ci_cov(row):
    n = int(row.draws); k = int(round(float(row.coverage_kg) * n))
    lo, hi = binomtest(k, n).proportion_ci(confidence_level=.95, method='wilson')
    return float(lo), float(hi)


def design_axis(ax, state, show_labels):
    y = np.arange(len(DESIGNS))
    for sep in GROUPS:
        ax.axhline(sep, color=MUTED, lw=.7, zorder=0)
    ax.set_yticks(y, [l for _, l in DESIGNS] if show_labels else [''] * len(y)); ax.set_ylim(len(y) - .45, -.55)


def ext_mse(state):
    def build(fig, ss, ctx):
        if ctx['standalone']:
            g = ss.subgridspec(2, 1, height_ratios=[.11, 1], hspace=.06); legend_strip(fig, g[0], target_handles(), 2); ss = g[1]
        ax = fig.add_subplot(ss); g = PRIMARY[PRIMARY.population.eq(state)]
        design_axis(ax, state, state == 'MI' or ctx['standalone'])
        for y, (p, _) in enumerate(DESIGNS):
            v = {t: float(g[g.policy.eq(p) & g.estimand.eq(t)].mse_ratio_to_SRS_Hajek.iloc[0]) for t in ['systems', 'people']}
            ax.plot([v['systems'], v['people']], [y, y], color=MUTED, lw=1.1, zorder=1)
            for t in ['systems', 'people']:
                ax.plot(v[t], y, ls='', marker=MK[t], ms=7, mfc=COL[t], mec=INK, mew=.6, zorder=3)
        ax.axvline(1, color=INK, ls=(0, (3, 3)), lw=.85, zorder=0)
        ax.set_xscale('log'); ax.set_xlim(.012, 6.5); ax.set_xticks([.02, .1, 1, 5])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}')); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(STATES[state], fontsize=12.5, pad=7)
        sty(ax, 'MSE / SRS–Hájek')
    return build


def ext_cov(state):
    def build(fig, ss, ctx):
        if ctx['standalone']:
            g = ss.subgridspec(2, 1, height_ratios=[.11, 1], hspace=.06); legend_strip(fig, g[0], target_handles(), 2); ss = g[1]
        ax = fig.add_subplot(ss); g = PRIMARY[PRIMARY.population.eq(state)]
        design_axis(ax, state, state == 'MI' or ctx['standalone'])
        for y, (p, _) in enumerate(DESIGNS):
            for t, off in [('systems', -.14), ('people', .14)]:
                row = g[g.policy.eq(p) & g.estimand.eq(t)].iloc[0]; v = float(row.coverage_kg); lo, hi = ci_cov(row)
                ax.errorbar(v * 100, y + off, xerr=[[(v - lo) * 100], [(hi - v) * 100]], fmt=MK[t], ms=6, mfc=COL[t], mec=INK,
                            mew=.55, ecolor=INK, elinewidth=.9, capsize=1.8, zorder=3)
        ax.axvline(95, color=INK, lw=.85, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(68, 100); ax.set_xticks([70, 80, 90, 100])
        if ctx['standalone']:
            ax.set_title(STATES[state], fontsize=12.5, pad=7)
        sty(ax, 'Coverage (%)')
    return build


def target_handles():
    return [Line2D([], [], ls='', marker=MK[t], mfc=COL[t], mec=INK, mew=.6, ms=7, label=l)
            for t, l in [('systems', 'System count'), ('people', 'Service population')]]


def layout7(fig):
    head = fig.add_axes([.03, .945, .96, .045]); head.axis('off')
    head.legend(handles=target_handles(), ncol=2, loc='center', handlelength=1.3, handletextpad=.4, columnspacing=1.4, borderpad=0)
    g = fig.add_gridspec(2, 2, width_ratios=[1.36, 1], left=.045, right=.985, bottom=.06, top=.925, wspace=.05, hspace=.20)
    return [g[0, 0], g[0, 1], g[1, 0], g[1, 1]]


# ----------------------------------------------------------------------------------------------- graphical abstract
def graphical_abstract():
    import workflow_schematic as ws        # author-drawn vector workflow, identical to Fig. 3a
    W_mm, H_mm = 186, 100
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4))
    ax = fig.add_axes([.0, .34, 1., .66]); ax.axis('off')
    ws.draw_workflow(ax, PAL, INK, kx=(.66 * H_mm) / W_mm, y0=.17, y1=.985, fs=10)
    for x, lab in [(.128, 'Monitored larger\nsystems'), (.392, 'Risk and error\nmodels'), (.660, 'Allocation and\nsampling depth'),
                   (.894, 'Systems and\npeople served')]:
        ax.text(x, .055, lab, ha='center', va='center', fontsize=11, color=INK, linespacing=1.1)
    for x0, x1 in [(.252, .271), (.535, .551), (.775, .793)]:
        ax.annotate('', xy=(x1, .55), xytext=(x0, .55), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.0))
    tiles = [('4,450', 'small systems estimated\nabove PFOA/PFOS MCLs'), ('1.9 M', 'people served by\nthose systems'),
             ('≈1,900', 'affected systems missed\nby larger-system models'),
             ('70–83%', 'less population error\nwith shorter schedules')]
    for i, (big, small) in enumerate(tiles):
        x0 = .012 + i * .247
        t = fig.add_axes([x0, .025, .235, .27]); t.axis('off')
        t.add_patch(Rectangle((0, 0), 1, 1, transform=t.transAxes, fc=[COL['assisted'], COL['systems'], COL['post'], COL['model_only']][i],
                              ec=INK, lw=.8, alpha=.55))
        t.text(.5, .66, big, ha='center', va='center', fontsize=17, color=INK, transform=t.transAxes)
        t.text(.5, .26, small, ha='center', va='center', fontsize=9.5, color=INK, transform=t.transAxes, linespacing=1.15)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / f'Graphical_Abstract.{ext}', dpi=600, facecolor='white',
                    metadata={'Author': 'CHONG LIU', 'Creator': 'CHONG LIU', 'Producer': 'CHONG LIU'})
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    if '5' in ARGS.only:
        layout.save_modules([national_counts, weight_sensitivity, selection_odds, butterfly], 'Fig_5', 'Fig_5', layout5,
                            figsize_mm=(190, 196), sizes_mm=[(185, 76), (105, 85), (105, 85), (185, 88)])
    if '7' in ARGS.only:
        layout.save_modules([ext_mse('MI'), ext_mse('WA'), ext_cov('MI'), ext_cov('WA')], 'Fig_7', 'Fig_7', layout7,
                            figsize_mm=(190, 170), sizes_mm=[(120, 106)] * 4)
    if 'GA' in ARGS.only:
        graphical_abstract()
    print('Rendered into', FIG)
