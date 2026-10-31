"""Frozen external-design Fig.7 and completion/inference Fig.S6.

Usage: python render_method_figures.py [--only main|supplement|all]
All inputs are existing frozen by-setting/by-seed tables; no model is fitted.
"""
from pathlib import Path
import sys,os,json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullLocator

from strengthening import paths
pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--data-root',type=Path);pre.add_argument('--output-root',type=Path);pre.add_argument('--qa-root',type=Path)
location_args,remaining_args=pre.parse_known_args();sys.argv=[sys.argv[0],*remaining_args]
P=Path(__file__).resolve().parent;ROOT=paths.WORKDIR
DATA=location_args.data_root.resolve() if location_args.data_root else paths.METHOD_RESULTS
HELPERS=P/'figure_helpers/figkit'
sys.path.insert(0,str(HELPERS));sys.path.insert(0,str(HELPERS/'publication_helpers'))
import figstyle as fs
import publication_layout as layout
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,export_subfigures,save_figure_with_subfigures
from scientific_palettes import get_palette
from audit_palette_usage import validate_series_colors
from matplotlib import pyplot as plt

DEL=location_args.output_root.resolve() if location_args.output_root else paths.WORKDIR;FIG=DEL/'figures' if location_args.output_root else paths.FIGURES
QA=location_args.qa_root.resolve() if location_args.qa_root else paths.WORKDIR/'figure_QA_regular';QA.mkdir(parents=True,exist_ok=True)
FDS=QA/'figure_source_data';FDS.mkdir(parents=True,exist_ok=True)
fs.OUTPUT=DEL;fs.FIG=FIG;fs.QA=QA;fs.MIN_PT=12;fs.LETTER_PT=12
layout.OUTPUT=DEL;layout.FIG=FIG
configure_publication_style()
plt.rcParams.update({'font.size':12,'font.weight':'normal','axes.labelsize':13.5,'axes.labelweight':'normal','axes.titleweight':'normal','xtick.labelsize':12.5,'ytick.labelsize':12.5,'legend.fontsize':12,'legend.title_fontsize':12,'axes.linewidth':1.,'lines.linewidth':1.4,'savefig.dpi':600})
PAL=get_palette('nature_new');INK=fs.INK
POLICIES=['srs_full_hajek','neyman5_full_difference','pps_full_hajek','system_anchor_two_phase','uniform20','dual_full','pps_two_phase_hajek']
LABELS=['SRS full','Neyman† full','PPS full','System 20%','Dual 20%','Dual full','PPS 20%']
SHORT_LABELS=['1 SRS full','2 Neyman†','3 PPS full','4 System 20%','5 Dual 20%','6 Dual full','7 PPS 20%']
COL={'systems':PAL[4],'people':PAL[3],
     'srs_full_hajek':PAL[6],'neyman5_full_difference':PAL[7],'pps_full_hajek':PAL[0],
     'system_anchor_two_phase':PAL[3],'uniform20':PAL[4],'dual_full':PAL[1],'pps_two_phase_hajek':PAL[2],
     'uniform40':PAL[5],'neyman_cost20':PAL[2],'range_cost20':PAL[0],
     'MI':PAL[1],'WA':PAL[0],'fitted':PAL[3],'auc-match':PAL[4],
     'wald':PAL[1],'wald_t':PAL[3],'logit_t':PAL[4],'wilson':PAL[0],'wilson_df':PAL[6],'kg':PAL[7]}
validate_series_colors('nature_new',COL)
PRIMARY=pd.read_csv(DATA/'external_primary_R12_B300.csv')
ALL=pd.read_csv(DATA/'external_design_complete_v2.csv')
CALENDAR=pd.read_csv(DATA/'external_calendar_R12_B100.csv')
FRESH=pd.read_csv(DATA/'range_newseeds10_14/completion_small_by_seed.csv')
HASHES={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'external_primary_R12_B300.csv',DATA/'external_design_complete_v2.csv',DATA/'external_calendar_R12_B100.csv',DATA/'range_newseeds10_14/completion_small_by_seed.csv']}
MK={'systems':'o','people':'D'}

def style(ax,x=None,y=None):
    style_axis(ax,xlabel=x,ylabel=y,tick_size=12.5,label_size=13.5)
    ax.tick_params(length=4.5,width=1,pad=3)
    for s in ['left','bottom']:ax.spines[s].set_linewidth(1.)
    return ax

def legend_strip(fig,ss,handles,ncol=2,**kw):
    ax=fig.add_subplot(ss);ax.axis('off')
    opts={'handlelength':1.1,'handletextpad':.4,'columnspacing':1.3,'borderpad':0};opts.update(kw)
    ax.legend(handles=handles,loc='center',ncol=ncol,**opts)
    return ax

def target_handles():
    return [Line2D([],[],ls='',marker=MK[t],mfc=COL[t],mec=INK,mew=.6,ms=6,label=l) for t,l in [('systems','Systems'),('people','Service population')]]

def ci_coverage(row):
    n=int(row.draws);k=int(round(float(row.coverage_kg)*n))
    lo,hi=binomtest(k,n).proportion_ci(confidence_level=.95,method='wilson')
    return float(lo),float(hi)

def value_format(v):return f'{v:.2f}' if v>=.1 else f'{v:.3f}'

def policy_handles():
    return [Line2D([],[],ls='',marker='o',mfc=COL[p],mec=INK,mew=.6,ms=5.5,label=l) for p,l in zip(POLICIES,SHORT_LABELS)]

def state_target_handles():
    return [Line2D([],[],ls='',marker=MK[t],mfc=COL[st],mec=INK,mew=.6,ms=5.2,label=f'{st} '+('sys.' if t=='systems' else 'pop.')) for st in ['MI','WA'] for t in ['systems','people']]

def main_axis(fig,ss,ctx,coverage=False):
    if ctx['standalone']:
        g=ss.subgridspec(2,1,height_ratios=[.20,1],hspace=.12)
        legend_strip(fig,g[0],policy_handles(),4,columnspacing=.5)
        ss=g[1]
    if coverage:
        g=ss.subgridspec(2,1,height_ratios=[.12,1],hspace=.20 if ctx['standalone'] else .035)
        handles=[Line2D([],[],ls='',marker=MK[t],mfc=PAL[4],mec=INK,mew=.6,ms=5,label=l) for t,l in [('systems','Sys.'),('people','Pop.')]]
        legend_strip(fig,g[0],handles,2,columnspacing=.7)
        ss=g[1]
    return fig.add_subplot(ss)

def main_mse(state,target):
    def build(fig,ss,ctx):
        ax=main_axis(fig,ss,ctx);g=PRIMARY[PRIMARY.population.eq(state)&PRIMARY.estimand.eq(target)&~PRIMARY.coefficient.eq('one')].set_index('policy')
        for i,p in enumerate(POLICIES):
            v=float(g.loc[p,'mse_ratio_to_SRS_Hajek'])
            ax.plot(i+1,v,ls='',marker='o',ms=6,mfc=COL[p],mec=INK,mew=.65,zorder=3)
        ax.axhline(1,color=INK,ls=(0,(3,3)),lw=.85,zorder=0)
        ax.set_xticks(range(1,8));ax.set_xlim(.45,7.55)
        if target=='systems':ax.set_ylim(0,4.7);ax.set_yticks([0,2,4])
        else:
            ax.set_yscale('log');ax.set_ylim(.014,1.6);ax.set_yticks([.02,.1,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.yaxis.set_minor_locator(NullLocator())
        if target=='systems' or ctx['standalone']:ax.set_title('Michigan (n = 1,283)' if state=='MI' else 'Washington (n = 855)',fontsize=12.5,pad=8)
        style(ax,None,('System MSE\n/ SRS–Hájek' if target=='systems' else 'Population MSE\n/ SRS–Hájek') if state=='MI' or ctx['standalone'] else None)
    return build

def main_coverage(state):
    def build(fig,ss,ctx):
        ax=main_axis(fig,ss,ctx,True);g=PRIMARY[PRIMARY.population.eq(state)&~PRIMARY.coefficient.eq('one')]
        for i,p in enumerate(POLICIES):
            for target,off in [('systems',-.12),('people',.12)]:
                row=g[g.policy.eq(p)&g.estimand.eq(target)].iloc[0];v=row.coverage_kg;lo,hi=ci_coverage(row)
                ax.errorbar(i+1+off,v,yerr=[[v-lo],[hi-v]],fmt=MK[target],ms=4.8,mfc=COL[p],mec=INK,mew=.55,ecolor=COL[p],elinewidth=1.25,capsize=1.8,zorder=3)
        ax.axhline(.95,color=INK,lw=.85,ls=(0,(3,3)),zorder=0)
        ax.set_ylim(.67,1.005);ax.set_yticks([.7,.8,.9,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v*100:.0f}'))
        ax.set_xticks(range(1,8));ax.set_xlim(.45,7.55)
        if ctx['standalone']:ax.set_title('Michigan' if state=='MI' else 'Washington',fontsize=12.5,pad=8)
        style(ax,'Design','KG coverage\n(%)' if state=='MI' or ctx['standalone'] else None)
    return build

def grid_three(fig,wspace=.18):
    g=fig.add_gridspec(3,2,left=.085,right=.99,bottom=.055,top=.88,wspace=wspace,hspace=.16)
    return [g[r,c] for r in range(3) for c in range(2)]

def main_grid(fig):
    head=fig.add_axes([.03,.90,.96,.055]);head.axis('off')
    head.legend(handles=policy_handles(),ncol=4,loc='center',handlelength=.7,handletextpad=.3,columnspacing=.6,borderpad=0)
    return grid_three(fig,wspace=.08)

def supplement_grid(fig):
    g=fig.add_gridspec(3,2,left=.085,right=.99,bottom=.055,top=.985,wspace=.18,hspace=.16)
    return [g[r,c] for r in range(3) for c in range(2)]

def pseudo_ratios(target):
    d=FRESH[FRESH.budget.eq(1200)&FRESH.estimand.eq(target)&FRESH.coefficient.eq('tuned')&FRESH.policy.isin(['uniform20','uniform40','neyman_cost20','range_cost20'])].copy()
    ref=d[d.policy.eq('uniform20')][['scenario','seed','mse']].rename(columns={'mse':'reference_mse_uniform20'})
    return d.merge(ref,on=['scenario','seed'],validate='many_to_one').assign(relative_mse=lambda x:x.mse/x.reference_mse_uniform20)

def pseudo_mse(target):
    def build(fig,ss,ctx):
        gs=ss.subgridspec(2,1,height_ratios=[.13,1],hspace=.04)
        handles=[Line2D([],[],ls='',marker=mk,mfc=COL[sc],mec=INK,mew=.6,ms=5,label=lab) for sc,mk,lab in [('fitted','o','Fitted'),('auc-match','D','AUC-matched')]]
        legend_strip(fig,gs[0],handles,2,columnspacing=.8)
        ax=fig.add_subplot(gs[1]);d=pseudo_ratios(target)
        for i,p in enumerate(['uniform20','uniform40','neyman_cost20','range_cost20']):
            for scen,off,mk in [('fitted',-.12,'o'),('auc-match',.12,'D')]:
                vals=d[d.policy.eq(p)&d.scenario.eq(scen)].relative_mse.to_numpy();v=vals.mean();lo=vals.min();hi=vals.max()
                ax.errorbar(i+off,v,yerr=[[v-lo],[hi-v]],fmt=mk,ms=5.5,mfc=COL[scen],mec=INK,mew=.6,ecolor=COL[scen],elinewidth=1.3,capsize=1.8,zorder=3)
        ax.axhline(1,color=INK,lw=.8,ls=(0,(3,3)),zorder=0)
        ax.set_xticks(range(4),['20%','40%','Neyman','Range']);ax.set_xlim(-.5,3.5)
        ax.set_ylim(.88,1.27);ax.set_yticks([.9,1,1.1,1.2]);style(ax,None,'System MSE\n/ uniform 20%' if target=='systems' else 'Population MSE\n/ uniform 20%')
        ax.tick_params(axis='x',labelrotation=28)
        for tick in ax.get_xticklabels():tick.set_ha('right')
    return build

def interval_frontier(fig,ss,ctx):
    ax=fig.add_subplot(ss)
    g=FRESH[FRESH.budget.eq(1200)&FRESH.scenario.eq('fitted')&FRESH.estimand.eq('systems')&FRESH.policy.eq('uniform20')&FRESH.coefficient.eq('tuned')]
    # Labels use fixed non-data positions and leaders; the observed points are never shifted.
    specs=[('wald','Wald',(5.43,90.7),'o'),('wald_t','Wald-t',(5.66,90.7),'s'),('logit_t','Logit-t',(5.91,92.7),'D'),('wilson','Wilson',(5.39,93.9),'^'),('wilson_df','Wilson-df',(5.65,95.2),'v'),('kg','KG',(5.91,94.1),'P')]
    for key,label,txt,mk in specs:
        x=g['width_'+key].mean()*100;y=g['coverage_'+key].mean()*100
        ax.plot(x,y,ls='',marker=mk,ms=8,mfc=COL[key],mec=INK,mew=.65,zorder=4)
        ax.annotate(label,(x,y),xytext=txt,textcoords='data',ha='center',va='center',fontsize=12,arrowprops={'arrowstyle':'-','lw':.75,'color':INK},bbox={'boxstyle':'square,pad=.08','fc':'white','ec':'none'},zorder=5)
    ax.axhline(95,color=INK,lw=.9,ls=(0,(3,3)),zorder=0)
    if ctx['standalone']:ax.set_title('Fitted scenario · systems',fontsize=12.5,pad=8)
    ax.set_xlim(5.30,6.01);ax.set_ylim(89.9,96.1);ax.set_xticks([5.4,5.7,6]);ax.set_yticks([90,93,96]);style(ax,'Width (percentage points)','Coverage (%)')

def information_mse(fig,ss,ctx):
    gs=ss.subgridspec(2,1,height_ratios=[.24,1],hspace=.10);legend_strip(fig,gs[0],state_target_handles(),2,columnspacing=.4,handlelength=.65,handletextpad=.25)
    ax=fig.add_subplot(gs[1]);regimes=['operational_learned_error','operational_sd_proxy','cold_region_sd_proxy']
    g=ALL[ALL.rule.eq('R12')&ALL.budget_full_equivalents.eq(300)&ALL.policy.eq('uniform20')&ALL.coefficient.eq('tuned')]
    for j,state in enumerate(['MI','WA']):
        for i,reg in enumerate(regimes):
            for ti,target in enumerate(['systems','people']):
                r=g[g.population.eq(state)&g.regime.eq(reg)&g.estimand.eq(target)].iloc[0]
                ax.plot(i+[-.21,-.07,.07,.21][j*2+ti],r.mse_ratio_to_SRS_Hajek,ls='',marker=MK[target],ms=5.2,mfc=COL[state],mec=INK,mew=.55,zorder=3)
    ax.axhline(1,color=INK,lw=.8,ls=(0,(3,3)),zorder=0);ax.set_yscale('log');ax.set_ylim(.018,2.7);ax.set_yticks([.03,.1,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.yaxis.set_minor_locator(NullLocator());ax.set_xlim(-.5,2.5)
    ax.set_xticks(range(3),['Learned','Proxy','Excluded']);style(ax,None,'MSE /\nSRS–Hájek')
    ax.tick_params(axis='x',labelrotation=28)
    for tick in ax.get_xticklabels():tick.set_ha('right')

def calendar_coverage(fig,ss,ctx):
    gs=ss.subgridspec(2,1,height_ratios=[.24,1],hspace=.10);legend_strip(fig,gs[0],state_target_handles(),2,columnspacing=.4,handlelength=.65,handletextpad=.25)
    ax=fig.add_subplot(gs[1]);pols=['srs_full_hajek','neyman5_full_difference','uniform20','pps_two_phase_hajek']
    for j,state in enumerate(['MI','WA']):
        g=CALENDAR[CALENDAR.population.eq(state)&~CALENDAR.coefficient.eq('one')]
        for i,p in enumerate(pols):
            for ti,target in enumerate(['systems','people']):
                r=g[g.policy.eq(p)&g.estimand.eq(target)].iloc[0];v=r.coverage_kg;lo,hi=ci_coverage(r)
                ax.errorbar(i+[-.21,-.07,.07,.21][j*2+ti],v,yerr=[[v-lo],[hi-v]],fmt=MK[target],ms=4.8,mfc=COL[state],mec=INK,mew=.55,ecolor=COL[state],elinewidth=1.2,capsize=1.8,zorder=3)
    ax.axhline(.95,color=INK,lw=.8,ls=(0,(3,3)),zorder=0);ax.set_ylim(.79,1.005);ax.set_yticks([.8,.9,1]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v*100:.0f}'));ax.set_xlim(-.5,3.5)
    ax.set_xticks(range(4),['SRS','Neyman†','Dual 20%','PPS 20%']);style(ax,None,'KG coverage\n(%)')
    ax.tick_params(axis='x',labelrotation=28)
    for tick in ax.get_xticklabels():tick.set_ha('right')

def correction_points():
    e=PRIMARY[PRIMARY.policy.isin(['system_anchor_two_phase','uniform20','system20','neyman_cost20','range_cost20','pps_two_phase_hajek'])&PRIMARY.estimand.eq('systems')&~PRIMARY.coefficient.eq('one')][['population','policy','coverage_kg','zero_missed_full_fraction']].rename(columns={'population':'group'})
    e['point_scope']='observed external finite cohort'
    z=FRESH[FRESH.budget.eq(1200)&FRESH.policy.isin(['uniform20','uniform40','neyman_cost20','range_cost20'])&FRESH.estimand.eq('systems')&FRESH.coefficient.eq('tuned')].groupby(['scenario','policy'])[['coverage_kg','zero_missed_full_fraction']].mean().reset_index().rename(columns={'scenario':'group'})
    z['point_scope']='mean over five new pseudo-populations'
    return pd.concat([e,z],ignore_index=True)

def correction_scatter(fig,ss,ctx):
    gs=ss.subgridspec(2,1,height_ratios=[.24,1],hspace=.10)
    spec=[('MI','MI','o'),('WA','WA','D'),('fitted','Fitted','^'),('auc-match','AUC-matched','s')]
    legend_strip(fig,gs[0],[Line2D([],[],ls='',marker=mk,mfc=COL[k],mec=INK,mew=.6,ms=5.2,label=l) for k,l,mk in spec],2,columnspacing=.7)
    ax=fig.add_subplot(gs[1]);d=correction_points()
    for k,label,mk in spec:
        g=d[d.group.eq(k)];ax.scatter(g.zero_missed_full_fraction*100,g.coverage_kg*100,s=30,marker=mk,c=COL[k],edgecolor=INK,linewidth=.6,zorder=3)
    ax.axhline(95,color=INK,lw=.9,ls=(0,(3,3)),zorder=0)
    ax.set_xlim(62,99);ax.set_ylim(80,98);ax.set_xticks([65,80,95]);ax.set_yticks([80,90,95]);style(ax,'No captured miss (%)','KG coverage\n(%)')

def save_source_data():
    PRIMARY[PRIMARY.policy.isin(POLICIES)&~PRIMARY.coefficient.eq('one')].to_csv(FDS/'Fig_7_external_monitoring.csv',index=False)
    pd.concat([pseudo_ratios('systems'),pseudo_ratios('people')],ignore_index=True).to_csv(FDS/'Fig_S6_ab_new_pseudo_populations.csv',index=False)
    selected=FRESH[FRESH.budget.eq(1200)&FRESH.scenario.eq('fitted')&FRESH.estimand.eq('systems')&FRESH.policy.eq('uniform20')&FRESH.coefficient.eq('tuned')]
    rows=[{'interval':k,'coverage':selected['coverage_'+k].mean(),'width':selected['width_'+k].mean(),'five_pseudo_population_mean':True} for k in ['wald','wald_t','logit_t','wilson','wilson_df','kg']]
    pd.DataFrame(rows).to_csv(FDS/'Fig_S6_c_interval_frontier.csv',index=False)
    ALL[ALL.rule.eq('R12')&ALL.budget_full_equivalents.eq(300)&ALL.policy.eq('uniform20')&ALL.coefficient.eq('tuned')].to_csv(FDS/'Fig_S6_d_information_conditions.csv',index=False)
    CALENDAR[CALENDAR.policy.isin(POLICIES)&~CALENDAR.coefficient.eq('one')].to_csv(FDS/'Fig_S6_e_calendar_coverage.csv',index=False)
    correction_points().to_csv(FDS/'Fig_S6_f_zero_correction.csv',index=False)
    (FDS/'Fig_7_S6_data_dictionary.md').write_text('''# Figure source data

MSE ratios use a cost-matched SRS–Hájek denominator within the same cohort, estimand and budget. No MSE uncertainty interval is drawn. `shared_sample_for_two_estimands=false` identifies target-specific Neyman references; they are not a joint design.

Fig. 7: count-window source-verified v2; R12; B=300 full-schedule equivalents; frozen operational f/u. Main policy codes and labels are in the plot script.

Fig. S6 a,b: paired MSE ratios to uniform20 within each new pseudo-population, scenario and target; B=1200; five generation seeds 10–14. Points are seed means, whiskers the minimum and maximum, not confidence intervals. c: five-seed mean coverage/width for fitted systems, uniform20, B=1200. d: same first-stage strategy and q20 with three frozen information conditions, B=300. e: different calendar-aligned finite cohorts at B=100; no raw error comparison with the primary cohorts. f: only corrected two-phase strategies, system target; full-schedule policies are excluded because they have no missed-event correction by construction. Cross-cohort points are diagnostic and do not define a causal trend.

Coverage bars are 95% Wilson binomial Monte Carlo intervals for 1,000 independent design draws; they quantify simulation precision, not uncertainty in the external cohort truth. Source cohort membership and operational predictions were frozen before design replay.
''',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['main','supplement','all'],default='all');a=ap.parse_args()
    save_source_data()
    if a.only in ['main','all']:
        builders=[main_mse('MI','systems'),main_mse('WA','systems'),main_mse('MI','people'),main_mse('WA','people'),main_coverage('MI'),main_coverage('WA')]
        layout.save_modules(builders,'Fig_7','Fig_7',main_grid,figsize_mm=(190,200),sizes_mm=[(180,100)]*6)
    if a.only in ['supplement','all']:
        builders=[pseudo_mse('systems'),pseudo_mse('people'),interval_frontier,information_mse,calendar_coverage,correction_scatter]
        layout.save_modules(builders,'Fig_S6','Fig_S6',supplement_grid,figsize_mm=(190,200),sizes_mm=[(180,100)]*6)
    assert HASHES=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'external_primary_R12_B300.csv',DATA/'external_design_complete_v2.csv',DATA/'external_calendar_R12_B100.csv',DATA/'range_newseeds10_14/completion_small_by_seed.csv']}
    (QA/'source_and_palette_audit.json').write_text(json.dumps({'source_hashes_unchanged':HASHES,'palette':'nature_new','series_colors':COL,'main_policies':POLICIES,'native_minimum_font_pt':12,'main_layout':'three content rows, two state columns','MSE_intervals':'none drawn','coverage_bars':'95% Wilson binomial Monte Carlo intervals for R1000','Neyman_reference':'target specific, no joint point'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
