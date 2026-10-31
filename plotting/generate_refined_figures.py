"""Publication composites: one scientific question per module, unequal-sized layouts.

Figure-only revision: frozen observations/models/simulations are unchanged.
Composition references inspected: HodgeMap current submission Figs 1, 2, 4, 6,
and its live fig1.py and fig6_global_atlas_v6.py. All panels redraw independently.
"""
from pathlib import Path
import argparse, json, sys
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from f_common import *
import figstyle as fs
import f01_fig1_monitoring_gap as f1
import f02_fig2_transfer_bias as f2
import f03_fig3_anchor_method as f3
import f04_fig4_census_validation as f4
import f05_fig5_small_burden as f5
import f06_fig6_external_validation as f6
import f07_fig7_next_round_design as f7
import f08_supporting_figures as si
import refine_overview_round3 as ov3
import refine_panels_round3 as rp3
import refine_mechanism_round3 as rm3
import refine_three_rows_a as tr4a
import refine_three_rows_b as tr4b
sys.stdout.reconfigure(encoding='utf-8')

fs.MIN_PT=12
fs.LETTER_PT=12
plt.rcParams.update({'font.size':12,'axes.labelsize':13.5,'axes.labelweight':'normal',
 'xtick.labelsize':12.5,'ytick.labelsize':12.5,'legend.fontsize':12,'legend.title_fontsize':12,
 'lines.linewidth':1.6,'axes.linewidth':1.0,'savefig.dpi':600})

def style(ax,xlabel=None,ylabel=None):
    style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=10.5,label_size=11.5)
    ax.tick_params(length=3.4,width=.8,pad=3)
    return ax

def polish(builder):
    """Keep the exact source plotting values; normalize visible labels and strokes."""
    def draw(fig,ss,ctx):
        before=set(fig.findobj(Axes));builder(fig,ss,ctx)
        for ax in [a for a in fig.findobj(Axes) if a not in before]:
            for obj in [ax.xaxis.label,ax.yaxis.label]:
                t=obj.get_text().replace('Small systems above MCL (%)','Exceeding systems (%)').replace('Systems above MCL (%)','Exceeding systems (%)').replace('People above MCL (thousands)','Affected population (thousands)').replace('Above MCL (%)','Exceedance share (%)')
                if obj is ax.yaxis.label:t=t.replace('People served per system','System size (people)')
                obj.set_text(t);obj.set_fontsize(11.5)
            for text in ax.findobj(Text):
                if text.get_text()=='Higher AUC':text.set_text('AUC matched')
            for ln in ax.lines:
                if ln.get_linestyle() not in ['None','none','',None] and ln.get_color() not in [INK,MUTED]:ln.set_linewidth(max(1.45,ln.get_linewidth()))
    return draw

def two_by_two(fig):
    g=fig.add_gridspec(2,2,left=.065,right=.99,top=.985,bottom=.055,wspace=.20,hspace=.20)
    return [g[0,0],g[0,1],g[1,0],g[1,1]]

def map_status(ax,zoom=False):
    basemap(ax,lw=.28,extent=not zoom)
    for nm,key,s,ec,lw in f1.CATS:
        d=f1.P[f1.P.category==nm].dropna(subset=['lat','lon'])
        map_points(ax,d.lon.to_numpy(),d.lat.to_numpy(),s=(7 if zoom else 5) if key=='sampled' else (2.3 if zoom else .85),color=C[key],edgecolor=INK if key=='sampled' else 'none',linewidths=.32 if key=='sampled' else 0,zorder=4 if key=='sampled' else 2)

def coverage_map(fig,ss,ctx):
    g=ss.subgridspec(2,2,height_ratios=[1,.12],width_ratios=[3.4,1],wspace=.055,hspace=.055)
    ax=fig.add_subplot(g[0,0]);map_status(ax)
    zoom=g[0,1].subgridspec(3,1,hspace=.25)
    for j,name in enumerate(['Northeast corridor','Carolinas–Georgia','Southern California']):
        z=fig.add_subplot(zoom[j]);map_status(z,True)
        x0,x1,y0,y1=fs.zoom_bbox(name);xm=(x0+x1)/2;ym=(y0+y1)/2
        width=max(x1-x0,(y1-y0)*2.0);height=width/2.0
        z.set_xlim(xm-width/2,xm+width/2);z.set_ylim(ym-height/2,ym+height/2)
        z.set_aspect('equal');z.set_title(name,loc='left',fontsize=9,pad=2)
        for sp in z.spines.values():sp.set_visible(True);sp.set_color(MUTED);sp.set_linewidth(.7)
        ax.add_patch(Rectangle((xm-width/2,ym-height/2),width,height,fill=False,ec=INK,lw=.7,zorder=8))
    n=f1.P.category.value_counts()
    handles=[Line2D([],[],ls='',marker='o',mfc=C[k],mec=INK,mew=.35,ms=4.6,label=f'{lab} (n = {n[nm]:,})') for nm,k,lab in [
      ('Unmonitored','unmonitored','Unmonitored'),('Census (≥3,300)','census','Larger systems'),('Sampled small (<3,300)','sampled','Small sample')]]
    legend_strip(fig,g[1,:],handles,3,columnspacing=1,handletextpad=.25)

def coverage_by_size(fig,ss,ctx):
    ax=fig.add_subplot(ss);d=f1.COV;y=np.arange(len(d));off=.13
    for i,(key,col,lab,mk) in enumerate([('systems','share_systems','Systems','o'),('people','share_people','People served','s')]):
        yy=y+(i-.5)*2*off
        ax.hlines(yy,.25,d[col]*100,color=C[key],lw=1.3)
        ax.plot(d[col]*100,yy,ls='',marker=mk,ms=5,color=C[key],mec=INK,mew=.45,label=lab)
    ax.set_yticks(y,d.cls);ax.set_ylim(len(d)-.55,-.5);ax.set_xscale('log');ax.set_xlim(.25,125)
    ax.set_xticks([.3,1,3,10,30,100],['0.3','1','3','10','30','100']);ax.minorticks_off()
    ax.axhline(3.5,color=MUTED,lw=.7,ls='--');ax.legend(loc='upper right',labelspacing=.35,handlelength=1.2)
    style(ax,'Monitored in UCMR5 (%)','People served per system')

def layout1(fig):
    g=fig.add_gridspec(2,1,height_ratios=[1.25,1],left=.065,right=.99,top=.985,bottom=.06,hspace=.065)
    low=g[1].subgridspec(1,2,width_ratios=[1,1.05],wspace=.17)
    return [g[0],low[0],low[1]]

def class_bias(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[f2.STRIP,1],hspace=.05)
    handles=[Line2D([],[],ls='',marker=m,mfc=C[c],mec=INK,mew=.5,ms=5,label=l) for m,c,l in [('o','observed','Design estimate'),('D','plugin','Predicted')]]
    legend_strip(fig,g[0],handles,1,labelspacing=.3)
    ax=fig.add_subplot(g[1]);d=f2.CLS;y=np.arange(len(d))
    for i,r in enumerate(d.itertuples()):
        ax.plot([r.plugin*100,r.obs*100],[i,i],color=MUTED,lw=1)
        ax.errorbar(r.obs*100,i,xerr=[[(r.obs-r.lo)*100],[(r.hi-r.obs)*100]],fmt='o',ms=5.8,mfc=C['observed'],mec=INK,mew=.5,ecolor=INK,elinewidth=.8,capsize=2)
        ax.plot(r.plugin*100,i,ls='',marker='D',ms=5,mfc=C['plugin'],mec=INK,mew=.5)
    ax.set_yticks(y,d.cls);ax.set_ylim(len(d)-.5,-.5);ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20])
    style(ax,'Exceeding systems (%)','People served per system')

def transfer_curve(fig,ss,ctx):
    f2.panel_c(fig,ss,ctx)
    ax=fig.axes[-1]
    ax.axvspan(1.4,np.log10(3300),color=C['sampled'],alpha=.09,lw=0,zorder=0)
    ax.text(.02,.96,'Small systems',ha='left',va='top',transform=ax.transAxes,fontsize=9)
    ax.text(.58,.96,'Larger systems',ha='left',va='top',transform=ax.transAxes,fontsize=9)
    style(ax,'People served per system','Exceeding systems (%)')

def layout2(fig):
    g=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.065,right=.99,top=.985,bottom=.06,wspace=.15,hspace=.085)
    return [g[0,0],g[0,1],g[1,:]]

def workflow(fig,ss,ctx):
    ax=fig.add_subplot(ss);ax.axis('off');ax.set_gid('schematic')
    asset=OUTPUT/'figures/Fig3_anchor_method/source/workflow_landscape.png'
    ax.imshow(plt.imread(asset),extent=[0,1,.14,.98],aspect='auto')
    names=[(.15,'Monitored larger systems'),(.435,'Risk and error models'),(.685,'Allocation +\nsampling depth'),(.892,'Systems +\npeople served')]
    for x,t in names:ax.text(x,.055,t,ha='center',va='center',fontsize=9)
    for x0,x1 in [(.296,.318),(.548,.572),(.785,.806)]:
        ax.annotate('',xy=(x1,.55),xytext=(x0,.55),arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.0))
    ax.set_xlim(0,1);ax.set_ylim(0,1)

def equation_strip(fig,ss,ctx):
    ax=fig.add_subplot(ss);ax.set_gid('schematic');ax.axis('off')
    terms=[r'$\lambda\sum_U a_i f_i$',r'$\sum_s\frac{a_i(\widetilde y_i-\lambda f_i)}{\pi_i}$',r'$\sum_{s_2}\frac{a_i(y_i-\widetilde y_i)}{\pi_i q}$']
    xs=[.27,.54,.83]
    ax.text(.005,.56,r'$\widehat\theta=\frac{1}{A}\,[\,$',fontsize=14,ha='left',va='center')
    for x,t,c,label in zip(xs,terms,[C['plugin'],C['recal'],C['ratio']],['Model total','Weighted residuals','Random-completion correction']):
        ax.text(x,.59,t,fontsize=12,ha='center',va='center')
        ax.plot([x-.105,x+.105],[.24,.24],color=c,lw=2.3)
        ax.text(x,.08,label,fontsize=9,ha='center',va='center')
    for x in [.397,.688]:ax.text(x,.57,'+',ha='center',va='center',fontsize=15)
    ax.text(.98,.57,']',ha='right',va='center',fontsize=17)
    ax.set_xlim(0,1);ax.set_ylim(0,1)

def allocation_composition(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.42,1],hspace=.06)
    handles=[Patch(fc=C[f'cls{i}'],ec=INK,lw=.4,label=lab) for i,lab in enumerate(CLASS_ORDER)]
    legend_strip(fig,g[0],handles,2,title='People served per system',handlelength=1.1,labelspacing=.25)
    ax=fig.add_subplot(g[1]);d=f3.PI
    groups=['<=100','101-500','501-1000','1001-3300'];rows=[]
    for col,lab in [('pi_epa','Reconstructed cells'),('pi_anchor','ANCHOR'),('pi_dual','Dual design')]:
        v=d.groupby('scls')[col].sum().reindex(groups)/d[col].sum()*100;rows.append(v.values)
    arr=np.array(rows);left=np.zeros(3)
    for j in range(4):
        ax.barh(range(3),arr[:,j],left=left,height=.6,color=C[f'cls{j}'],ec=INK,lw=.45)
        for y,l,v in zip(range(3),left,arr[:,j]):
            if v>=12:ax.text(l+v/2,y,f'{v:.0f}',ha='center',va='center',fontsize=9)
        left+=arr[:,j]
    ax.set_yticks(range(3),['Cells','ANCHOR','Dual']);ax.set_ylim(2.5,-.5);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
    style(ax,'Expected sample composition (%)')
    pd.DataFrame(arr,index=['Reconstructed cells','ANCHOR','Dual design'],columns=CLASS_ORDER).to_csv(FD/'fig3_expected_composition.csv')

def layout3(fig):
    g=fig.add_gridspec(3,1,height_ratios=[1,.44,1.25],left=.055,right=.99,top=.99,bottom=.055,hspace=.045)
    b=g[2].subgridspec(1,2,width_ratios=[1,1.05],wspace=.18)
    return [g[0],g[1],b[0],b[1]]

def layout4(fig):
    g=fig.add_gridspec(4,1,height_ratios=[.14,1,1,.88],left=.065,right=.99,top=.99,bottom=.045,hspace=.065)
    legend_strip(fig,g[0],f4.method_handles(),4,handlelength=1.7,columnspacing=.85,labelspacing=.3)
    top=g[1].subgridspec(1,2,wspace=.16);mid=g[2].subgridspec(1,2,wspace=.16)
    low=g[3].subgridspec(1,2,width_ratios=[1,1.1],wspace=.15)
    return [top[0],top[1],mid[0],mid[1],low[0],low[1]]

def compact_coverage(fig,ss,ctx):
    f4.panel_coverage(fig,ss,ctx);ax=fig.axes[-1]
    names={'Population PPS':'PPS','ML-assisted GREG':'GREG','SRS + difference':'SRS + diff.','Neyman allocation':'Neyman','Neyman + difference':'Neyman + diff.'}
    ax.set_yticks(ax.get_yticks(),[names.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()])

def compact_ablation(fig,ss,ctx):
    f4.panel_ablation(fig,ss,ctx);ax=fig.axes[-1]
    ax.set_yticks(range(len(f4.ABL)),['ANCHOR','τ = 0.25','τ = 0.75','λ = 1','Fixed n','SD score','GREG-lin','Hájek','Uniform'])
    ax.set_xlabel('Efficiency vs SRS')

def states_lollipop(fig,ss,ctx):
    ax=fig.add_subplot(ss);d=f5.S[f5.S.ST.str.len()==2].sort_values('people_exc',ascending=False).head(12);y=np.arange(len(d))
    ax.hlines(y,0,d.people_exc/1000,color=C['recal'],lw=2.6)
    ax.errorbar(d.people_exc/1000,y,xerr=[(d.people_exc-d.people_exc_lo)/1000,(d.people_exc_hi-d.people_exc)/1000],fmt='o',ms=5,mfc=C['recal'],mec=INK,mew=.45,ecolor=INK,elinewidth=.8,capsize=1.8)
    ax.set_yticks(y,d.ST);ax.set_ylim(len(d)-.4,-.6);ax.set_xlim(0,310);ax.set_xticks([0,100,200,300]);style(ax,'Affected population (thousands)')

def layout5(fig):
    g=fig.add_gridspec(3,1,height_ratios=[1.28,.08,1],left=.065,right=.99,top=.99,bottom=.055,hspace=.065)
    top=g[0].subgridspec(1,2,width_ratios=[1.7,1],wspace=.12)
    legend_strip(fig,g[1],f5.est_handles(),3,columnspacing=1.4)
    low=g[2].subgridspec(1,2,width_ratios=[1.08,1],wspace=.18)
    return [top[0],top[1],low[0],low[1]]

def prevalence_facets(fig,ss,ctx):
    g=ss.subgridspec(1,2,wspace=.16)
    for j,(st,name) in enumerate(f6.STATES):
        ax=fig.add_subplot(g[j]);truth=f6.FU.loc[f'{st}: state truth (tested small systems)','value']*100
        ax.axvline(truth,color=INK,lw=.9,ls=(0,(3,2)))
        for i,(key,lab,mk,col) in enumerate(f6.ESTS):
            r=f6.FU.loc[f'{st}: {key}']
            if np.isfinite(r.lo):ax.plot([r.lo*100,r.hi*100],[i,i],color=INK,lw=.9)
            ax.plot(r.value*100,i,ls='',marker=mk,ms=6,color=C[col],mec=INK,mew=.5)
        ax.set_yticks(range(4),['Model','Ratio','Logit','Additive'] if j==0 else ['']*4);ax.set_ylim(3.55,-.8);ax.set_xlim(0,8);ax.set_xticks([0,2,4,6,8])
        style(ax,'Exceeding systems (%)')
        ax.text(.02,.98,name,ha='left',va='top',transform=ax.transAxes,fontsize=12)
        ax.text(.98,.97,f'Observed: {truth:.2f}%',ha='right',va='top',transform=ax.transAxes,fontsize=9)

def auc_forest(fig,ss,ctx):
    f6.panel_auc(fig,ss,ctx);ax=fig.axes[-1];ax.set_xlim(.4,1.035)
    for i,s in enumerate(f6.SETS):
        r=f6.AUC[f6.AUC.dataset==s].iloc[0]
        ax.text(1.025,i,f'{r.auc:.2f}',ha='right',va='center',fontsize=9)
    ax.spines['bottom'].set_bounds(.4,1.0)

def recall(fig,ss,ctx):
    f6.panel_recall(fig,ss,ctx);ax=fig.axes[-1]
    ax.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100']);ax.set_yticks([0,.25,.5,.75,1],['0','25','50','75','100'])
    style(ax,'Systems tested (%)','Exceedances found (%)')

def layout6(fig):
    g=fig.add_gridspec(2,1,height_ratios=[.85,1.15],left=.065,right=.99,top=.985,bottom=.055,hspace=.085)
    low=g[1].subgridspec(1,2,width_ratios=[1.08,1],wspace=.14)
    return [g[0],low[0],low[1]]

def rules(fig,ss,ctx):
    f7.panel_rules(fig,ss,ctx);ax=fig.axes[-1]
    for rule in ['R1','R12','R2']:
        d=f7.W[f7.W.rule==rule].sort_values('miss_rate');ax.plot(d.cost_saved*100,d.miss_rate*100,color=f7.RC[rule],lw=1.1,zorder=2)

def depth_eff(fig,ss,ctx):
    f7.panel_eff(fig,ss,ctx);ax=fig.axes[-1]
    ax.set_yticks(range(6),['Equal π: R12','ANCHOR: full','R12','R12 + 2-phase','R1','R1 + 2-phase'])
    ax.set_xticks([0,1,2,3,4,5])
    style(ax,'Relative efficiency')
    # Compact legend; no long handle padding in a half-width panel.
    for axis in fig.axes:
        lg=axis.get_legend()
        if lg is not None and not axis.axison:
            handles=lg.legend_handles;labels=[t.get_text() for t in lg.get_texts()];lg.remove()
            axis.legend(handles,labels,ncol=3,loc='center',handlelength=.7,handletextpad=.3,columnspacing=.6)

def projected_cost(fig,ss,ctx):
    f7.panel_small(fig,ss,ctx);ax=fig.axes[-1]
    ax.set_yticks(range(6),['Cells: full','Cells: R12 + correction','ANCHOR: full','ANCHOR: R12 + correction','ANCHOR: R1 + correction','Dual: R12 + correction'])
    ax.set_xlabel('Projected analytical samples')
    lg=ax.get_legend()
    if lg:
        for t in lg.get_texts():
            if t.get_text()=='Higher AUC':t.set_text('AUC matched')
    # The removed duplicate cost panel contributed ranges for the shared design.
    # Preserve them here for exactly the corresponding two scenarios and targets.
    d=shared_design_summary()
    for j,e in enumerate(['systems','people']):
        yy=5+(j-.5)*.38
        for scenario,dy in [('fitted',-.055),('auc-match',.055)]:
            v=d[(d.scenario==scenario)&(d.estimand==e)].iloc[0]
            ax.errorbar(v.need_mean,yy+dy,xerr=[[v.need_mean-v.need_min],[v.need_max-v.need_mean]],fmt='none',ecolor=INK,elinewidth=.8,capsize=2,zorder=4)

def shared_design_summary():
    d=si.SM[si.SM.strategy=='Dual · R12 + two-phase'].copy()
    refs=si.SM[si.SM.strategy.str.startswith('REFERENCE')].set_index(['scenario','seed','estimand']).mse
    d=d.join(refs.rename('mse_ref'),on=['scenario','seed','estimand'])
    d['need']=d.cost_mean*d.mse/d.mse_ref
    summary=d.groupby(['scenario','estimand']).agg(need_mean=('need','mean'),need_min=('need','min'),need_max=('need','max'),coverage_mean=('cov_kg','mean'),coverage_min=('cov_kg','min'),coverage_max=('cov_kg','max')).reset_index()
    summary.to_csv(OUTPUT/'checks/shared_design_summary.csv',index=False)
    return summary

def shared_coverage_forest(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.18,1],hspace=.02)
    handles=[Line2D([],[],ls='',marker=m,mfc=C[e],mec=INK,mew=.5,ms=5,label=label) for e,m,label in [('systems','s','Systems'),('people','D','People served')]]
    legend_strip(fig,g[0],handles,2)
    ax=fig.add_subplot(g[1]);d=shared_design_summary()
    rows=[('fitted','systems','Fitted: systems'),('fitted','people','Fitted: people'),('auc-match','systems','AUC matched: systems'),('auc-match','people','AUC matched: people')]
    for y,(scenario,e,label) in enumerate(rows):
        r=d[(d.scenario==scenario)&(d.estimand==e)].iloc[0]
        ax.errorbar(r.coverage_mean,y,xerr=[[r.coverage_mean-r.coverage_min],[r.coverage_max-r.coverage_mean]],fmt='s' if e=='systems' else 'D',ms=5.5,mfc=C[e],mec=INK,mew=.5,ecolor=INK,elinewidth=.9,capsize=2)
    ax.set_yticks(range(4),[r[2] for r in rows]);ax.set_ylim(3.6,-.6);ax.set_xlim(.915,.965);ax.set_xticks([.92,.94,.95,.96]);ax.axvline(.95,color=MUTED,ls='--',lw=.8)
    style(ax,'95% interval coverage')

def depth_coverage(fig,ss,ctx):
    handles=[Line2D([],[],color=f7.RC[r],lw=1.5,ls=ls,marker=m,ms=4,mfc=f7.RC[r] if ls=='-' else 'white',mec=INK,mew=.5,label=l) for _,l,r,ls,m in f7.C_LINES]
    g=ss.subgridspec(2,1,height_ratios=[.27,1],hspace=.04)
    legend_strip(fig,g[0],handles,2,handlelength=1.3,columnspacing=.8,labelspacing=.25)
    ax=fig.add_subplot(g[1]);ns=[150,300,600]
    for j,(strategy,lab,rule,ls,marker) in enumerate(f7.C_LINES):
        values=f7.CV[f7.CV.strategy==strategy].groupby('n0').cov_kg
        x=np.arange(3)+(j-2)*.07;means=values.mean().reindex(ns);lo=values.min().reindex(ns);hi=values.max().reindex(ns)
        ax.vlines(x,lo,hi,color=f7.RC[rule],lw=1)
        ax.plot(x,means,color=f7.RC[rule],lw=1.5,ls=ls,marker=marker,ms=4.5,mfc=f7.RC[rule] if ls=='-' else 'white',mec=INK,mew=.45)
    ax.axhline(.95,color=INK,lw=.7,ls=':');ax.set_ylim(.4,.98);ax.set_yticks([.4,.6,.8,.95]);ax.set_xlim(-.35,2.35);ax.set_xticks(range(3),ns)
    style(ax,'Budget (system equivalents)','95% interval coverage')

def layout7(fig):
    g=fig.add_gridspec(3,1,height_ratios=[1,1.1,.57],left=.065,right=.99,top=.99,bottom=.04,hspace=.08)
    top=g[0].subgridspec(1,2,width_ratios=[1,1.38],wspace=.13)
    middle=g[1].subgridspec(1,2,width_ratios=[1,1.38],wspace=.13)
    return [top[0],top[1],middle[0],middle[1],g[2]]

def shared_panel(builder):
    def draw(fig,ss,ctx):
        before=set(fig.axes);builder(fig,ss,ctx)
        if not ctx['standalone']:
            for ax in [a for a in fig.axes if a not in before]:
                if not ax.axison and ax.get_legend() is not None:fig.delaxes(ax)
                elif ax.get_legend() is not None:ax.get_legend().remove()
    return draw

def interval_width(fig,ss,ctx):
    si.interval_panel(True)(fig,ss,ctx);ax=fig.axes[-1];ax.set_ylim(.99,1.08);ax.set_yticks([1,1.02,1.04,1.06,1.08])

def supplement1(fig):
    g=fig.add_gridspec(4,2,height_ratios=[.1,1,.1,1],left=.065,right=.99,top=.99,bottom=.055,wspace=.14,hspace=.065)
    legend_strip(fig,g[0,:],[Line2D([],[],color=c,marker='o',ms=4,label=lab) for _,lab,c in si.IV],4)
    legend_strip(fig,g[2,:],[Line2D([],[],color=c,marker='o',ms=4,label=str(n)) for c,n in zip([PAL[1],PAL[3],PAL[4]],[150,300,600])],3,title='Budget (full-schedule system equivalents)')
    return [g[1,0],g[1,1],g[3,0],g[3,1]]

HEAT=LinearSegmentedColormap.from_list('white_teal',['white',PAL[3]])
def sensitivity_heatmap(est):
    def draw(fig,ss,ctx):
        from matplotlib.colors import Normalize
        d=si.SP[(si.SP.n==800)&(si.SP.estimand==est)];sc=['fitted','auc-match','fitted+sd'];mu=[];sd=[]
        for de,ee,l,c in si.DES:
            group=d[(d.design==de)&(d.estimator==ee)].groupby('scenario').eff_vs_epa
            mu.append(group.mean().reindex(sc).values);sd.append(group.std().reindex(sc).values)
        mu=np.array(mu);sd=np.array(sd);ax=fig.add_subplot(ss)
        ax.imshow(mu,cmap=HEAT,vmin=0,vmax=5,aspect='auto',interpolation='nearest')
        for i in range(6):
            for j in range(3):ax.text(j,i,f'{mu[i,j]:.2f}\n±{sd[i,j]:.2f}',ha='center',va='center',fontsize=9,linespacing=1.08)
        ax.set_yticks(range(6),['SRS','Cells','Neyman','Neyman + diff.','ANCHOR','Dual'])
        ax.set_xticks(range(3),['Fitted','AUC\nmatched','Extra\nvariation'])
        style(ax,ylabel='Systems' if est=='systems' else 'People served')
        ax.set_xticks(np.arange(-.5,3),minor=True);ax.set_yticks(np.arange(-.5,6),minor=True);ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0)
        if ctx['standalone']:
            cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,5),cmap=HEAT),ax=ax,orientation='horizontal',fraction=.05,pad=.15)
            cb.set_ticks([0,1,3,5]);cb.set_label('Efficiency vs reconstructed cells',fontsize=9)
    return draw

def supplement2(fig):
    from matplotlib.colors import Normalize
    g=fig.add_gridspec(5,2,height_ratios=[1.12,.22,.12,.72,1.23],left=.065,right=.99,top=.99,bottom=.045,wspace=.14,hspace=.065)
    cax=fig.add_subplot(g[1,:]);pos=cax.get_position();cax.set_position([pos.x0+.20*pos.width,pos.y0+.7*pos.height,pos.width*.60,pos.height*.28])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,5),cmap=HEAT),cax=cax,orientation='horizontal');cb.set_ticks([0,1,3,5]);cb.set_label('Efficiency vs reconstructed cells',fontsize=9,labelpad=2)
    hs=[Line2D([],[],ls='',marker=m,color=c,mec=INK,mew=.5,label=l,ms=5) for c,m,l in [(PAL[1],'s','Geography'),(PAL[3],'D','+ Neighbours')]]
    legend_strip(fig,g[2,:],hs,2)
    return [g[0,0],g[0,1],g[3,0],g[3,1],g[4,:]]

def supplement3(fig):
    g=fig.add_gridspec(2,2,height_ratios=[.1,1],left=.065,right=.99,top=.99,bottom=.07,wspace=.18,hspace=.1)
    legend_strip(fig,g[0,:],[Patch(fc=C[e],ec=INK,lw=.45,label=l) for e,l in [('systems','Systems'),('people','People served')]],2)
    return [g[1,0],g[1,1]]

def feature_dumbbell(est):
    def draw(fig,ss,ctx):
        hs=[Line2D([],[],ls='',marker=m,color=c,mec=INK,mew=.5,label=l,ms=5) for c,m,l in [(PAL[1],'s','Geography'),(PAL[3],'D','+ Neighbours')]]
        if ctx['standalone']:
            g=ss.subgridspec(2,1,height_ratios=[.22,1],hspace=.05)
            legend_strip(fig,g[0],hs,2)
            ax=fig.add_subplot(g[1])
        else:ax=fig.add_subplot(ss)
        keys=[('Neyman-H5 + PPI++ (hybrid)','Neyman: 5 strata'),('Neyman-H10 + PPI++ (hybrid)','Neyman: 10 strata'),('ANCHOR (Poisson + PPI++)','ANCHOR')]
        d=si.G[(si.G.n==300)&(si.G.estimand==est)&(si.G.population=='medium')]
        for i,(key,lab) in enumerate(keys):
            vals=[d[(d.prior==p)&(d.label==key)].rel_eff.iloc[0] for p in ['geo','geo+nb']]
            ax.plot(vals,[i,i],color=MUTED,lw=1.0)
            for v,c,m in zip(vals,[PAL[1],PAL[3]],['s','D']):ax.plot(v,i,ls='',marker=m,color=c,mec=INK,mew=.5,ms=6)
        ax.set_yticks(range(3),[l for k,l in keys]);ax.set_ylim(2.7,-.6);ax.set_xlim(.9,2.5);ax.set_xticks([1,1.5,2,2.5]);ax.axvline(1,color=MUTED,ls='--',lw=.8)
        style(ax,'Efficiency: '+('systems' if est=='systems' else 'people served'))
    return draw

def save(num,builders,layout,size,sizes,folder):
    save_modules([polish(b) for b in builders],folder,folder,layout,figsize_mm=size,sizes_mm=sizes)

TASKS={
 '1':([ov3.coverage_map3,ov3.coverage_by_size3,tr4a.small_composition4],tr4a.layout1_three_rows,(190,194),[(195,95),(180,85),(180,60)],'Fig1_monitoring_gap'),
 '2':([rm3.roc3,rm3.class_bias3,rm3.transfer_curve3],tr4a.layout2_three_rows,(190,194),[(180,85),(180,75),(185,70)],'Fig2_transfer_bias'),
 '3':([rm3.workflow3,tr4a.allocation4,tr4a.expected_composition4],tr4a.layout3_three_rows,(190,194),[(185,76),(180,80),(180,60)],'Fig3_anchor_method'),
 '4':([ov3.efficiency_round3('medium','systems',ylim=(0,2.65),yticks=[0,1,2]),ov3.efficiency_round3('large','systems',ylim=(0,2.35),yticks=[0,1,2]),ov3.efficiency_round3('medium','people',ylim=(0,3.15),yticks=[0,1,2,3]),ov3.efficiency_round3('large','people',logy=True,ylim=(.3,70),yticks=[.3,1,3,10,30]),tr4a.pareto4,ov3.coverage_round3],tr4a.layout4_three_rows,(190,184),[(105,86)]*4+[(100,90),(105,95)],'Fig4_census_validation'),
 '5':([f5.panel_map,states_lollipop,tr4b.national_facets4,rp3.size_forest3],tr4b.layout5_three_rows,(190,188),[(125,88),(90,95),(180,70),(180,65)],'Fig5_small_burden'),
 '6':([prevalence_facets,rp3.auc_forest3,tr4b.recall_wide4],tr4b.layout6_three_rows,(190,194),[(180,68),(180,65),(180,75)],'Fig6_external_validation'),
 '7':([tr4b.rules_scatter4,depth_eff,rp3.projected_cost3,rp3.depth_coverage3,rp3.shared_coverage3],tr4b.layout7_three_rows,(190,190),[(95,86),(115,86),(180,80),(105,90),(115,75)],'Fig7_next_round_design'),
 'S1':([shared_panel(si.interval_panel()),shared_panel(interval_width),shared_panel(rm3.fraction3()),shared_panel(rm3.fraction3(True))],rm3.supplement1_round3,(190,137),[(95,82)]*4,'FigS1_interval_completion'),
 'S2':([sensitivity_heatmap('systems'),sensitivity_heatmap('people'),shared_panel(feature_dumbbell('systems')),shared_panel(feature_dumbbell('people')),rp3.ablation_dots3],rm3.supplement2_round3,(190,192),[(100,96),(100,96),(100,73),(100,73),(180,70)],'FigS2_design_sensitivity')
}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--only',nargs='*');a=ap.parse_args()
    for number in a.only or TASKS:
        bs,lo,sz,subs,folder=TASKS[number];save(number,bs,lo,sz,subs,folder)
