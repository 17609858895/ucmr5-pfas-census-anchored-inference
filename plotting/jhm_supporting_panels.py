"""Readable uncertainty and sensitivity panels from the frozen summaries."""
from f_common import *
import f08_supporting_figures as source
import refine_mechanism_round3 as mechanics

SCENARIOS=[('fitted','Fitted',PAL[1],'o'),('auc-match','AUC matched',PAL[7],'s'),('fitted+sd','Extra variation',PAL[2],'^')]

def layout_s1(fig):
    g=fig.add_gridspec(4,2,height_ratios=[.14,1,.25,1],left=.065,right=.99,top=.99,bottom=.055,wspace=.18,hspace=.075)
    legend_strip(fig,g[0,:],[Line2D([],[],color=c,marker='o',ms=4,label=l) for _,l,c in source.IV],4,handlelength=1.05,columnspacing=1.0)
    legend_strip(fig,g[2,:],[Line2D([],[],color=c,marker='o',ms=4,label=str(n)) for c,n in zip([PAL[1],PAL[3],PAL[4]],[150,300,600])],3,title='Budget (full-schedule system equivalents)',handlelength=1.05,labelspacing=.3)
    return [g[1,0],g[1,1],g[3,0],g[3,1]]

def interval_coverage(fig,ss,ctx):
    source.interval_panel()(fig,ss,ctx)
    for ax in fig.axes:
        if ax.axison and ax.get_ylabel()=='95% interval coverage':ax.set_ylabel('Coverage')

def interval_width(fig,ss,ctx):
    source.interval_panel(True)(fig,ss,ctx);ax=fig.axes[-1]
    ax.set_ylim(.99,1.08);ax.set_yticks([1,1.02,1.04,1.06,1.08]);ax.set_ylabel('Width / Wald')

def fraction(eff=False):
    builder=mechanics.fraction3(eff)
    def draw(fig,ss,ctx):
        builder(fig,ss,ctx)
        ax=fig.axes[-1];ax.set_ylabel('Efficiency / full' if eff else 'Coverage')
    return draw

def scenario_forest(target):
    def draw(fig,ss,ctx):
        if ctx['standalone']:
            g=ss.subgridspec(2,1,height_ratios=[.18,1],hspace=.07)
            legend_strip(fig,g[0],[Line2D([],[],color=c,marker=m,ms=5,mec=INK,mew=.4,label=label) for _,label,c,m in SCENARIOS],3,handlelength=.7,columnspacing=.8)
            ss=g[1]
        ax=fig.add_subplot(ss);d=source.SP[(source.SP.n==800)&(source.SP.estimand==target)]
        for j,(scenario,_,color,marker) in enumerate(SCENARIOS):
            vals=[];sds=[]
            for design,estimator,_,_ in source.DES:
                v=d[(d.design==design)&(d.estimator==estimator)&(d.scenario==scenario)].eff_vs_epa
                vals.append(v.mean());sds.append(v.std())
            y=np.arange(6)+(j-1)*.2
            ax.errorbar(vals,y,xerr=sds,fmt=marker,ms=4.9,mfc=color,mec=INK,mew=.45,ecolor=color,elinewidth=1.05,capsize=2,zorder=3)
        ax.set_yticks(range(6),['SRS','Cell ref.','Neyman','Neyman + diff.','ANCHOR','Dual'])
        ax.set_ylim(5.55,-.55);ax.axvline(1,color=MUTED,lw=.8,ls='--',zorder=1)
        if target=='systems':ax.set_xlim(0,5);ax.set_xticks([0,1,2,3,4,5])
        else:ax.set_xlim(0,2.25);ax.set_xticks([0,.5,1,1.5,2])
        style_axis(ax,xlabel='Relative efficiency',ylabel='Systems' if target=='systems' else 'People served',tick_size=12.5,label_size=13.5)
    return draw

def layout_s2(fig):
    g=fig.add_gridspec(5,2,height_ratios=[.14,1.2,.14,.70,1.1],left=.065,right=.99,top=.99,bottom=.045,wspace=.20,hspace=.085)
    legend_strip(fig,g[0,:],[Line2D([],[],color=c,marker=m,ms=5,mec=INK,mew=.45,label=label) for _,label,c,m in SCENARIOS],3,handlelength=1.1,columnspacing=1.1)
    hs=[Line2D([],[],ls='',marker=m,color=c,mec=INK,mew=.5,label=l,ms=5) for c,m,l in [(PAL[1],'s','Geography'),(PAL[3],'D','+ Neighbours')]]
    legend_strip(fig,g[2,:],hs,2,handlelength=.7,columnspacing=1.2)
    return [g[1,0],g[1,1],g[3,0],g[3,1],g[4,:]]
