"""Three-row manuscript compositions for Figs 5--7.

Frozen result tables are unchanged.  The shared source builders remain the
authority for estimates, intervals, projections and coverage ranges.
"""
from pathlib import Path
from f_common import *
import figstyle as fs
import f05_fig5_small_burden as f5
import f06_fig6_external_validation as f6
import f07_fig7_next_round_design as f7
import refine_panels_round3 as r3


def _style(ax,xlabel=None,ylabel=None):
    style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=10.5,label_size=11.5)
    ax.tick_params(length=3.4,width=.8,pad=3)


def national_facets4(fig,ss,ctx):
    """Six original national estimates and all four original interval pairs."""
    g=ss.subgridspec(2,1,height_ratios=[.20,1],hspace=.04)
    legend_strip(fig,g[0],f5.est_handles(),3,columnspacing=1.15,handlelength=1.1)
    row=g[1].subgridspec(1,2,wspace=.17)
    for j,(estimand,xlabel) in enumerate([
        ('systems','Exceeding systems (%)'),
        ('people','Affected service population (%)')]):
        ax=fig.add_subplot(row[j])
        for y,(key,label,marker,color) in enumerate(f5.EST):
            r=f5.B[(f5.B.group=='all')&(f5.B.estimand==estimand)
                  &(f5.B.estimator==key)].iloc[0]
            if np.isfinite(r.kg_lo):
                ax.plot([r.kg_lo*100,r.kg_hi*100],[y,y],color=INK,lw=.9,zorder=2)
            ax.plot(r.theta*100,y,ls='',marker=marker,ms=5.5,color=C[color],
                    mec=INK,mew=.5,zorder=3)
        ax.set_yticks(range(3),[r[1] for r in f5.EST] if j==0 else ['']*3)
        ax.set_ylim(2.6,-.6);ax.set_xlim(0,13);ax.set_xticks([0,4,8,12])
        _style(ax,xlabel)


def layout5_three_rows(fig):
    """a,b geographic row; c national row; d size-specific row."""
    g=fig.add_gridspec(3,1,height_ratios=[1.20,.76,.72],left=.065,right=.99,
                        top=.99,bottom=.05,hspace=.09)
    top=g[0].subgridspec(1,2,width_ratios=[1.65,1],wspace=.14)
    return [top[0],top[1],g[1],g[2]]


def recall_wide4(fig,ss,ctx):
    """Original cumulative curves with a compact low-information legend."""
    f6.panel_recall(fig,ss,ctx);ax=fig.axes[-1]
    lg=ax.get_legend();handles=lg.legend_handles
    labels=[t.get_text() for t in lg.get_texts()];lg.remove()
    ax.legend(handles,labels,loc='lower right',ncol=2,handlelength=1.35,
              handletextpad=.35,columnspacing=1.0,labelspacing=.25,borderaxespad=.3)
    ax.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100'])
    ax.set_yticks([0,.25,.5,.75,1],['0','25','50','75','100'])
    _style(ax,'Systems tested (%)','Exceedances found (%)')


def layout6_three_rows(fig):
    """a state recalibration; b discrimination; c cumulative detection."""
    g=fig.add_gridspec(3,1,height_ratios=[.95,.98,1],left=.065,right=.99,
                        top=.985,bottom=.05,hspace=.095)
    return [g[0],g[1],g[2]]


def rules_scatter4(fig,ss,ctx):
    """Nine categorical size-class/rule points, without implied interpolation."""
    f7.panel_rules(fig,ss,ctx)
    _style(fig.axes[-1],'Samples saved (%)','Missed exceedances (%)')


def layout7_three_rows(fig):
    """a,b tradeoffs; c full-width cost; d,e interval-coverage rows."""
    g=fig.add_gridspec(3,1,height_ratios=[1.0,1.03,.94],left=.065,right=.99,
                        top=.99,bottom=.045,hspace=.085)
    top=g[0].subgridspec(1,2,width_ratios=[1,1.25],wspace=.16)
    low=g[2].subgridspec(1,2,width_ratios=[1.13,1],wspace=.18)
    return [top[0],top[1],g[1],low[0],low[1]]


def _drafts():
    import generate_refined_figures as base
    from publication_layout import draw_module
    import os
    os.environ['FIG_DPI']='200'
    fs.MIN_PT=10.5;fs.LETTER_PT=12
    plt.rcParams.update({'font.size':10.5,'axes.labelsize':11.5,
      'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,
      'legend.title_fontsize':10.5,'axes.labelweight':'normal'})
    out=OUTPUT/'内部核查'/'第四轮草稿';out.mkdir(parents=True,exist_ok=True)
    fs.QA=out/'qa'
    plans=[
      ('Fig5_three_rows',(190,188),
       [f5.panel_map,base.states_lollipop,national_facets4,r3.size_forest3],layout5_three_rows),
      ('Fig6_three_rows',(190,194),
       [base.prevalence_facets,r3.auc_forest3,recall_wide4],layout6_three_rows),
      ('Fig7_three_rows',(190,190),
       [rules_scatter4,base.depth_eff,r3.projected_cost3,r3.depth_coverage3,r3.shared_coverage3],layout7_three_rows)]
    for stem,size,builders,layout in plans:
        fig=plt.figure(figsize=np.array(size)/25.4)
        for builder,cell,letter in zip(builders,layout(fig),'abcde'):
            draw_module(fig,cell,letter,base.polish(builder))
        report=fs._export(fig,out,stem,True)
        print(stem,report['n_unexpected'],len(report['label_overlaps']),
              report['alignment_max_residual_pt'],flush=True)


if __name__=='__main__':_drafts()
