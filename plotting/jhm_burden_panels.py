"""Burden, weighting sensitivity, and additive size-domain composition.

All estimates are supplied by the independent statistical audit. No bootstrap,
model fitting, renormalization or re-estimation is performed by plotting.
"""
from f_common import *
import f05_fig5_small_burden as source
from matplotlib.legend_handler import HandlerTuple

SENS=pd.read_csv(FD/'weight_calibration_sensitivity.csv')
SHARE=pd.read_csv(FD/'fig5c_additive_domain_shares.csv')
DOMAIN=pd.read_csv(FD/'additive_domain_burden_canonical.csv')
ODDS=pd.read_csv(FD/'within_cell_selection_odds_curve.csv')
INVENTORY=pd.read_csv(FD/'fig5c_full_inventory_shares.csv')
TIMING=pd.read_csv(FD/'timing_burden_sensitivity.csv')
CLASSES=['<=100','101-500','501-1000','1001-3300']
LABELS=['≤100','101–500','501–1,000','1,001–3,300']
NATIONAL=[(key,label,marker,color) for (key,_,marker,color),label in zip(source.EST,['Post-stratified','Model-assisted','Model-only'])]

def sty(ax,xlabel=None,ylabel=None):
    style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=12.5,label_size=13.5)

def national(fig,ss,ctx):
    # Row labels already identify all three methods; a second colour legend
    # would duplicate that information in this paired display.
    row=ss.subgridspec(1,2,wspace=.14)
    for j,(target,xlabel) in enumerate([('systems','Exceeding systems (%)'),('people','Associated service population (%)')]):
        ax=fig.add_subplot(row[j])
        for y,(key,label,mk,col) in enumerate(NATIONAL):
            r=source.B[(source.B.group=='all')&(source.B.estimand==target)&(source.B.estimator==key)].iloc[0]
            if np.isfinite(r.kg_lo):
                ax.plot([r.kg_lo*100,r.kg_hi*100],[y,y],color=INK,lw=1,zorder=2)
            ax.plot(r.theta*100,y,ls='',marker=mk,ms=6,color=C[col],mec=INK,mew=.5,zorder=3)
        ax.set_yticks(range(3),[x[1] for x in NATIONAL] if j==0 else ['']*3)
        ax.set_ylim(2.6,-.6);ax.set_xlim(0,13);ax.set_xticks([0,4,8,12])
        sty(ax,xlabel if target=='systems' else 'Associated service\npopulation (%)')

SENS_INDICES=[0,1,2,3,6,7]
SENS_LABELS=['Equal','Source','Size','Source × size','+ Region','Exclude 3,300']
def weight_sensitivity(fig,ss,ctx):
    ax=fig.add_subplot(ss);d=SENS.iloc[SENS_INDICES].copy()
    values=[*list(d.systems_difference*100),TIMING.loc[TIMING.scheme=='span_le366_days','systems_difference'].iloc[0]*100]
    y=np.arange(len(values))
    # All alternatives are assumption scenarios, not independent estimates with
    # comparative sampling intervals. Primary is identified by its hollow ring.
    ax.axvline(d.loc[d.scheme=='Source-by-size (primary)','systems_difference'].iloc[0]*100,color=MUTED,lw=.8,ls='--',zorder=1)
    ax.plot(values,y,ls='',marker='o',ms=5.8,color=C['systems'],mec=INK,mew=.6,zorder=3)
    ax.plot(d.iloc[3].systems_difference*100,3,ls='',marker='o',ms=9,mfc='none',mec=INK,mew=.8,zorder=4)
    ax.set_yticks(y,SENS_LABELS+['Span ≤366 d']);ax.set_ylim(len(values)-.45,-.6)
    ax.set_xlim(6.5,8.3);ax.set_xticks([6.5,7,7.5,8])
    sty(ax,'Exceeding systems (%)')

def selection_odds(fig,ss,ctx):
    ax=fig.add_subplot(ss)
    ax.plot(ODDS.gamma,ODDS.system_prevalence*100,color=C['systems'],lw=1.7,zorder=3)
    r=ODDS.iloc[np.argmin(abs(ODDS.gamma-1))]
    ax.plot(r.gamma,r.system_prevalence*100,ls='',marker='o',ms=5.5,mfc=C['systems'],mec=INK,mew=.6,zorder=4)
    ax.axvline(1,color=MUTED,lw=.8,ls='--',zorder=1)
    ax.set_xscale('log');ax.minorticks_off();ax.set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
    ax.set_xlim(.25,4);ax.set_ylim(0,27);ax.set_yticks([0,10,20])
    sty(ax,'Exceedance-odds ratio\n(unmonitored / observed)','Exceeding systems (%)')

def composition(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.25,1],hspace=.08)
    inventory=tuple(Line2D([],[],ls='',marker='s',ms=5.5,mfc='white',mec=C[e],mew=1.1) for e in ['systems','people'])
    exceed=tuple(Line2D([],[],ls='',marker='o',ms=5.5,mfc=C[e],mec=INK,mew=.5) for e in ['systems','people'])
    la=fig.add_subplot(g[0]);la.axis('off')
    la.legend([inventory,exceed],['Inventory','Associated with exceedance'],handler_map={tuple:HandlerTuple(ndivide=None,pad=.2)},loc='center',ncol=2,handlelength=2.2,handletextpad=.4,columnspacing=1.25)
    row=g[1].subgridspec(1,2,wspace=.16)
    for j,(target,label) in enumerate([('systems','Share of systems (%)'),('people','Share of people served (%)')]):
        ax=fig.add_subplot(row[j]);d=SHARE[SHARE.estimand==target].set_index('class').loc[CLASSES]
        inventory_share=INVENTORY[INVENTORY.estimand==target].set_index('class').loc[CLASSES].inventory_share_pct
        for y,(_,r) in enumerate(d.iterrows()):
            ax.errorbar(r.share_pct,y-.09,xerr=[[r.share_pct-r.lo_pct],[r.hi_pct-r.share_pct]],fmt='o',ms=5.8,mfc=C[target],mec=INK,mew=.5,ecolor=INK,elinewidth=.9,capsize=2,zorder=3)
            ax.plot(inventory_share.iloc[y],y+.09,ls='',marker='s',ms=5.5,mfc='white',mec=C[target],mew=1.2,zorder=4)
        ax.set_yticks(range(4),LABELS if j==0 else ['']*4)
        ax.set_ylim(3.5,-.5);ax.set_xlim(0,80);ax.set_xticks([0,25,50,75])
        sty(ax,label,'System size (people)' if j==0 else None)

def layout5(fig):
    g=fig.add_gridspec(3,1,height_ratios=[.86,1.0,1.06],left=.065,right=.99,top=.99,bottom=.045,hspace=.12)
    middle=g[1].subgridspec(1,2,wspace=.20)
    return [g[0],middle[0],middle[1],g[2]]

def size_prevalence(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.19,1],hspace=.05)
    hs=[Line2D([],[],ls='',marker=mk,ms=5.5,mfc=C[col],mec=INK,mew=.5,label=label) for label,mk,col in [('Model-assisted','D','ANCHOR (Poisson + PPI++)'),('Model-only','s','plugin')]]
    legend_strip(fig,g[0],hs,2,handlelength=.65,columnspacing=1.2)
    ax=fig.add_subplot(g[1]);d=DOMAIN[DOMAIN.estimand=='systems'].set_index('size_band').loc[CLASSES]
    for y,(cls,r) in enumerate(d.iterrows()):
        p=source.B[(source.B.group==list(source.SIZE_KEY.values())[y])&(source.B.estimand=='systems')&source.B.estimator.str.startswith('ML plug-in')].iloc[0]
        ax.plot([p.theta*100,r.prevalence_pct],[y,y],color=MUTED,lw=.8,zorder=1)
        ax.errorbar(r.prevalence_pct,y,xerr=[[r.prevalence_pct-r.prevalence_lo_pct],[r.prevalence_hi_pct-r.prevalence_pct]],fmt='D',ms=5.8,mfc=C['ANCHOR (Poisson + PPI++)'],mec=INK,mew=.5,ecolor=INK,elinewidth=.9,capsize=2,zorder=3)
        ax.plot(p.theta*100,y,ls='',marker='s',ms=5.5,mfc=C['plugin'],mec=INK,mew=.5,zorder=3)
    ax.set_yticks(range(4),LABELS);ax.set_ylim(3.5,-.5)
    ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20])
    sty(ax,'Exceeding systems (%)','System size (people)')

def layout_s3(fig):
    g=fig.add_gridspec(3,1,height_ratios=[1.14,.80,1.0],left=.065,right=.99,top=.99,bottom=.045,hspace=.09)
    top=g[0].subgridspec(1,2,width_ratios=[1.3,1],wspace=.16)
    return [top[0],top[1],g[1],g[2]]

def state_map(fig,ss,ctx):
    source.panel_map(fig,ss,ctx)
    for ax in fig.findobj(Axes):
        if hasattr(ax,'_colorbar'):
            ax.set_xlabel('Projected exceedance (%)')

def state_population(fig,ss,ctx):
    import generate_refined_figures as base
    base.states_lollipop(fig,ss,ctx)
    sty(fig.axes[-1],'Associated service\npopulation (10³)')
