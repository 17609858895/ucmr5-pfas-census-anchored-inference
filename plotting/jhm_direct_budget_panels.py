"""Direct fixed-budget simulations, without inverse-MSE extrapolation."""
from f_common import *
D=pd.read_csv(FD/'direct_budget_figure_data.csv')
ROWS=[('fitted',1000,'Fitted: 1,000'),('fitted',1200,'Fitted: 1,200'),('auc-match',1000,'Matched: 1,000'),('auc-match',1200,'Matched: 1,200')]

def mse(fig,ss,ctx):
    row=ss.subgridspec(1,2,wspace=.17)
    for j,(target,label) in enumerate([('systems','System MSE ratio'),('people','Population MSE ratio')]):
        ax=fig.add_subplot(row[j])
        for y,(scenario,budget,_) in enumerate(ROWS):
            r=D[(D.scenario==scenario)&(D.budget==budget)&(D.estimand==target)].iloc[0]
            ax.errorbar(r.mean_mse_ratio,y,xerr=[[r.mean_mse_ratio-r.min_mse_ratio],[r.max_mse_ratio-r.mean_mse_ratio]],fmt='s' if target=='systems' else 'D',ms=6,mfc=C[target],mec=INK,mew=.5,ecolor=INK,elinewidth=.95,capsize=2,zorder=3)
        ax.set_yticks(range(4),[r[2] for r in ROWS] if j==0 else ['']*4)
        ax.set_ylim(3.5,-.5);ax.set_xlim(.6,1.42);ax.set_xticks([.6,.8,1,1.2,1.4],['0.6','0.8','1','1.2','1.4'])
        ax.axvline(1,color=MUTED,lw=.8,ls='--',zorder=1)
        style_axis(ax,xlabel=label,tick_size=12.5,label_size=13.5)

def coverage(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.18,1],hspace=.07)
    hs=[Line2D([],[],ls='',marker=mk,ms=5.5,mfc=C[target],mec=INK,mew=.5,label=label) for target,label,mk in [('systems','Systems','s'),('people','People','D')]]
    legend_strip(fig,g[0],hs,2,handlelength=.65,handletextpad=.3,columnspacing=.8)
    ax=fig.add_subplot(g[1])
    for y,(scenario,budget,_) in enumerate(ROWS):
        for target,mk,offset in [('systems','s',-.11),('people','D',.11)]:
            r=D[(D.scenario==scenario)&(D.budget==budget)&(D.estimand==target)].iloc[0]
            ax.errorbar(r.mean_coverage,y+offset,xerr=[[r.mean_coverage-r.min_coverage],[r.max_coverage-r.mean_coverage]],fmt=mk,ms=5.5,mfc=C[target],mec=INK,mew=.5,ecolor=INK,elinewidth=.9,capsize=2,zorder=3)
    ax.set_yticks(range(4),[r[2] for r in ROWS]);ax.set_ylim(3.55,-.55)
    ax.set_xlim(.91,.965);ax.set_xticks([.92,.94,.96],['0.92','0.94','0.96']);ax.axvline(.95,color=MUTED,lw=.8,ls='--',zorder=1)
    style_axis(ax,xlabel='Coverage',tick_size=12.5,label_size=13.5)
