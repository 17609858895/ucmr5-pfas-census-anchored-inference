from strengthening import paths
"""Main Fig.6 and SI Fig.S5 from frozen independent-state outcomes.

Native quantitative drawing; no generated or manually altered observations.
Uses the retained project publication kit for exact-palette, alignment,
typography, 600-dpi PNG/PDF and independently redrawn semantic panels.
"""
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd
from scipy.stats import binomtest
from matplotlib.patches import Patch,Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

P=paths.EXTERNAL_RESULTS; ROOT=paths.WORKDIR; O=paths.EXTERNAL_RESULTS
kit=paths.PLOTTING_CODE
sys.path.insert(0,str(kit));sys.path.insert(0,str(kit/'figkit'))
import publication_layout as layout
import figstyle as fs
from publication_layout import plt,style_axis
from matplotlib.text import Text

DEL=paths.WORKDIR
layout.FIG=paths.FIGURES;fs.QA=P/'figure_QA';fs.MIN_PT=12;fs.LETTER_PT=12
plt.rcParams.update({'font.size':12,'font.weight':'normal','axes.labelsize':13.5,'axes.labelweight':'normal',
 'xtick.labelsize':12.5,'ytick.labelsize':12.5,'legend.fontsize':12,'legend.title_fontsize':12,
 'lines.linewidth':1.6,'axes.linewidth':1.0,'savefig.dpi':600})

# User preference for this isolated revision: every figure Text uses regular
# weight. Apply before layout calibration and audit every object after export.
_source_export=fs._export
def export_regular_text(fig,folder,stem,letters_on):
    for obj in fig.findobj(Text):obj.set_fontweight('normal')
    report=_source_export(fig,folder,stem,letters_on)
    objects=fig.findobj(Text)
    bad=[obj.get_text() for obj in objects if obj.get_fontweight() not in ('normal',400)]
    assert not bad,(stem,bad)
    report['all_Text_objects_normal']=True
    report['all_Text_object_count']=len(objects)
    report['font_weight_inventory']=sorted({str(obj.get_fontweight()) for obj in objects})
    target=fs.QA/(stem+'.json');audit=json.loads(target.read_text(encoding='utf-8'))
    audit['style'].update(report)
    audit['text_objects']=[{'text':obj.get_text(),'weight':str(obj.get_fontweight()),'size_pt':obj.get_fontsize(),'visible':obj.get_visible()} for obj in objects]
    # Physical visual extents include all titles, ticks, labels and legends.
    # Recording these supports reproducible panel-gap comparisons without
    # changing the common alignment/export helper.
    fig.draw_without_rendering();renderer=fig.canvas.get_renderer()
    width,height=fig.get_size_inches()*25.4
    audit['compact_geometry']={
        'native_canvas_mm':[float(width),float(height)],
        'panels':[{'panel':r['key'],'outer_mm':(np.asarray(fs._panel_bbox(fig,r['axes'],renderer).extents)*[width,height,width,height]).tolist(),
            'axes_outer_mm':[(np.asarray(fs._panel_bbox(fig,[a],renderer).extents)*[width,height,width,height]).tolist() for a in r['axes'] if a.get_visible()]}
            for r in getattr(fig,'_panel_records',[])],
        'legend_strip_axes':sum(not a.axison and a.get_legend() is not None for a in fig.axes)}
    target.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return report
fs._export=export_regular_text
PAL=fs.get_palette('nature_new');INK=fs.INK
C={'observed':PAL[0],'frozen':PAL[1],'ratio':PAL[4],'available':PAL[1],'cold':PAL[4],'below4':PAL[1],'detect4':PAL[0]}
fs.validate_series_colors('nature_new',C)
CAL=pd.read_csv(O/'independent_cohort_calibration_transfer.csv')
M=pd.read_csv(O/'strict_external_metrics_bootstrap.csv')
SEL=pd.read_csv(O/'followup_eligibility_selection_summary.csv')
SENS=pd.read_csv(O/'threshold_duplicate_and_quantization_sensitivity.csv')
SCHED=pd.read_csv(O/'schedule_sensitivity_metrics.csv')
TRANS=pd.read_csv(O/'calendar_count_paired_transitions.csv')
STATES=[('MI','Michigan'),('WA','Washington')]

def style(ax,xlabel=None,ylabel=None):
    style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=12.5,label_size=13.5)
    ax.tick_params(length=4.5,width=1,pad=3)
    return ax

def state_facets(fig,ss,left_space=.16):
    g=ss.subgridspec(1,2,wspace=left_space)
    return [fig.add_subplot(g[0,i]) for i in range(2)]

def panel_prevalence(fig,ss,ctx):
    axes=state_facets(fig,ss,.16)
    for ax,(st,name) in zip(axes,STATES):
        rows=CAL[CAL.state.eq(st)];unc=rows[rows.prediction_method.eq('Uncorrected')].iloc[0]
        rat=rows[rows.prediction_method.eq('National ratio')].iloc[0]
        vals=np.array([unc.observed,unc.predicted,rat.predicted])*100
        ax.bar(np.arange(3),vals,color=[C['observed'],C['frozen'],C['ratio']],edgecolor=INK,linewidth=.9,width=.62,zorder=2)
        ax.errorbar(2,vals[2],yerr=[[100*(rat.predicted-rat.lo)],[100*(rat.hi-rat.predicted)]],fmt='none',ecolor=INK,elinewidth=1.2,capsize=3,zorder=3)
        for x,v in enumerate(vals):
            y=v+.35 if x!=2 else 100*rat.hi+.35
            ax.text(x,y,f'{v:.2f}',ha='center',va='bottom',fontsize=12)
        ax.set_xticks([0,1,2],['Observed','Frozen\nmodel','National\nratio']);ax.set_yticks([0,5,10]);ax.set_ylim(0,14)
        ax.set_title(f'{name} (n = {int(unc.n):,})',fontsize=12,pad=7)
        style(ax,ylabel='Cohort exceedance (%)' if st=='MI' else None)

def panel_auc(fig,ss,ctx):
    axes=state_facets(fig,ss,.16)
    models=[('f','Frozen\noriginal','s','frozen',False),('available_neighbours','Available\nneighbours','o','available',True),('cold_neighbours','Region\nexcluded','D','cold',False)]
    for ax,(st,name) in zip(axes,STATES):
        g=M[M.state.eq(st)&M.subset.eq('all')&M.resampling.eq('spatial_0.5degree')]
        ax.axvline(.5,color=INK,lw=.8,ls=(0,(3,3)),zorder=1)
        for i,(model,label,marker,key,filled) in enumerate(models):
            r=g[g.model.eq(model)].iloc[0]
            ax.plot([r.auc_lo,r.auc_hi],[i,i],color=INK,lw=1.15,zorder=2)
            ax.plot(r.auc,i,ls='',marker=marker,ms=7.0,mfc=C[key] if filled else 'white',mec=C[key],mew=1.8,zorder=3)
        ax.set_ylim(2.5,-.5);ax.set_yticks([0,1,2],[x[1] for x in models] if st=='MI' else ['', '', ''])
        ax.set_xlim(.35,.8);ax.set_xticks([.4,.5,.6,.7,.8]);ax.set_title(name,fontsize=12,pad=7)
        style(ax,xlabel='AUC')

def panel_selection(fig,ss,ctx):
    axes=state_facets(fig,ss,.16)
    for ax,(st,name) in zip(axes,STATES):
        r=SEL[SEL.state.eq(st)&SEL.unit.eq('system')&SEL.subgroup.eq('early_index')&SEL.initial_condition.eq('first_detect4')].iloc[0]
        for i,prefix,key in [(0,'negative','below4'),(1,'positive','detect4')]:
            n=int(r[prefix+'_n']);k=int(r[prefix+'_eligible']);p=k/n;ci=binomtest(k,n).proportion_ci(confidence_level=.95,method='exact')
            ax.bar(i,p*100,color=C[key],edgecolor=INK,linewidth=.9,width=.62,zorder=2)
            ax.errorbar(i,p*100,yerr=[[100*(p-ci.low)],[100*(ci.high-p)]],fmt='none',ecolor=INK,lw=1.15,capsize=3,zorder=3)
            ax.text(i,100*ci.high+3.6,f'{k:,}/{n:,}',ha='center',va='bottom',fontsize=12)
        ax.set_xticks([0,1],['<4 ng/L','≥4 ng/L']);ax.set_ylim(0,116);ax.set_yticks([0,50,100]);ax.set_title(name,fontsize=12,pad=7)
        style(ax,ylabel='Follow-up eligible (%)' if st=='MI' else None)

def main_layout(fig):
    g=fig.add_gridspec(3,1,left=.07,right=.99,top=.985,bottom=.035,hspace=.06,height_ratios=[1,1,1])
    return [g[i,0] for i in range(3)]

def endpoint_panel(state):
    def builder(fig,ss,ctx):
        ax=fig.add_subplot(ss);obs=[];pred=[];ns=[]
        r=M[M.state.eq(state)&M.model.eq('f')&M.subset.eq('all')&M.resampling.eq('system')].iloc[0]
        obs.append(r.observed*100);pred.append(r.predicted*100);ns.append(int(r.n))
        for rule in ['calendar_2023_2025','first_k_minimum_spacing']:
            r=SCHED[SCHED.state.eq(state)&SCHED.model.eq('f')&SCHED.resampling.eq('system')&SCHED.schedule.eq(rule)].iloc[0]
            obs.append(r.observed*100);pred.append(r.predicted*100);ns.append(int(r.n))
        x=np.arange(3)
        ax.bar(x-.18,obs,width=.34,color=C['observed'],edgecolor=INK,lw=.8,label='Observed')
        ax.bar(x+.18,pred,width=.34,color=C['frozen'],edgecolor=INK,lw=.8,label='Frozen model')
        for i,v,n in zip(x,obs,ns):ax.text(i,max(v,pred[i])+.5,f'n = {n:,}',ha='center',va='bottom',fontsize=12)
        ax.set_xticks(x,['Count','Calendar','Spaced']);ax.set_ylim(0,18.8);ax.set_yticks([0,5,10,15]);ax.set_title(dict(STATES)[state],fontsize=12,pad=7)
        ax.legend(loc='upper left',ncol=1,handlelength=.9,handletextpad=.35,borderaxespad=.15,labelspacing=.15)
        style(ax,ylabel='Cohort exceedance (%)' if state=='MI' or ctx['standalone'] else None)
    return builder

def transition_panel(state):
    def builder(fig,ss,ctx):
        ax=fig.add_subplot(ss);r=TRANS[TRANS.state.eq(state)].iloc[0]
        n01=int(r.became_positive);n10=int(r.became_negative);n11=int(r.count_positive_on_shared_locations)-n10;nn=int(r.shared_systems)
        tab=np.array([[nn-n01-n10-n11,n01],[n10,n11]])
        cmap=LinearSegmentedColormap.from_list('count_cells',['white',PAL[1]])
        ax.imshow(tab,cmap=cmap,vmin=0,vmax=max(tab.max(),1),aspect='auto',interpolation='nearest')
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor=INK,lw=.75))
                ax.text(j,i,f'{int(tab[i,j]):,}',ha='center',va='center',fontsize=13)
        ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Negative','Positive'])
        ax.set_title(f'{dict(STATES)[state]} (n = {nn:,})',fontsize=12,pad=7)
        style(ax,xlabel='Calendar endpoint',ylabel='Count endpoint' if state=='MI' or ctx['standalone'] else None)
        ax.tick_params(length=0)
    return builder

E=[('y_full','4.05 ng/L'),('y_threshold4','4.0 ng/L'),('y_PFOS_allmax','PFOS\nmaximum'),('y_PFOS_allsum','PFOS\nsum')]
def precision_panel(is_auc=False):
    def builder(fig,ss,ctx):
        ax=fig.add_subplot(ss);g=SENS[SENS.state.eq('WA')&SENS.model.eq('f')]
        for i,(end,label) in enumerate(E):
            r=g[g.endpoint.eq(end)].iloc[0];v=r.auc if is_auc else r.prevalence*100
            ax.plot(v,i,ls='',marker='o',mfc=PAL[0] if not is_auc else PAL[1],mec=INK,mew=.7,ms=6.5)
        scenario=g[g.endpoint.eq('hypothetical_nearest_integer_rounding_scenario_exact_AUC_range')].iloc[0]
        lo=scenario.auc_lower if is_auc else scenario.positives/scenario.n*100
        hi=scenario.auc_upper if is_auc else scenario.positive_upper/scenario.n*100
        ax.plot([lo,hi],[4,4],color=PAL[4],lw=3,solid_capstyle='butt')
        ax.plot([lo,hi],[4,4],ls='',marker='|',color=INK,ms=9,mew=1)
        ax.set_yticks(range(5),[r[1] for r in E]+['Rounding\nscenario'] if not is_auc or ctx['standalone'] else ['']*5)
        ax.set_ylim(4.65,-.55);ax.set_title('Washington',fontsize=12,pad=7)
        if is_auc:ax.set_xlim(.54,.64);ax.set_xticks([.55,.60,.65]);style(ax,xlabel='Frozen-score AUC')
        else:ax.set_xlim(8,11.5);ax.set_xticks([8,9,10,11]);style(ax,xlabel='Cohort exceedance (%)')
    return builder

def supp_layout(fig):
    g=fig.add_gridspec(3,2,left=.07,right=.99,top=.985,bottom=.035,wspace=.09,hspace=.07,height_ratios=[1,1,1.10])
    return [g[i,j] for i in range(3) for j in range(2)]

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['6','S5']);args=ap.parse_args()
    if args.only in [None,'6']:
        layout.save_modules([panel_prevalence,panel_auc,panel_selection],'Fig_6','Fig_6',main_layout,figsize_mm=(190,195),sizes_mm=[(190,66)]*3)
    if args.only in [None,'S5']:
        layout.save_modules([endpoint_panel('MI'),endpoint_panel('WA'),transition_panel('MI'),transition_panel('WA'),precision_panel(False),precision_panel(True)],'Fig_S5','Fig_S5',supp_layout,figsize_mm=(190,210),sizes_mm=[(115,82)]*6)
    # Exact displayed data travel with the statistical source, independent of plots.
    fd=P/'figure_data';fd.mkdir(exist_ok=True)
    CAL[CAL.prediction_method.isin(['Uncorrected','National ratio'])].to_csv(fd/'Fig_6a_calibration.csv',index=False)
    M[M.model.isin(['f','available_neighbours','cold_neighbours'])&M.subset.eq('all')&M.resampling.eq('spatial_0.5degree')].to_csv(fd/'Fig_6b_spatial_auc.csv',index=False)
    SEL[SEL.unit.eq('system')&SEL.subgroup.eq('early_index')&SEL.initial_condition.eq('first_detect4')].to_csv(fd/'Fig_6c_followup_eligibility.csv',index=False)
    SENS[SENS.state.eq('WA')&SENS.model.eq('f')].to_csv(fd/'Fig_S5_precision.csv',index=False)
    SCHED[SCHED.model.eq('f')&SCHED.resampling.eq('system')].to_csv(fd/'Fig_S5_schedule.csv',index=False)
    (P/'figure_specification.json').write_text(json.dumps({'palette':'nature_new, exact get_palette entries','main':{'path':'07/Figures/Fig_6','canvas_mm':[190,195],'semantic_panels':['a: two selected-cohort prevalence facets; national-factor-only ratio intervals','b: external discrimination, spatial 0.5-degree block-bootstrap intervals','c: first-detection follow-up eligibility with exact binomial intervals']},'supplement':{'path':'07/Figures/Fig_S5','canvas_mm':[190,210],'panels':['a,b: observed and frozen prediction means, selected schedule cohorts','c,d: paired exact shared-location endpoint transitions','e: WA threshold/duplicate prevalence points and hypothetical rounding scenario range','f: corresponding frozen-score AUC points and exact rank-based rounding scenario range']},'scenario_not_CI':'Rounding assumption is undocumented; purple ranges are conditional assumption scenarios, not sampling CIs or identified true-label bounds.','metadata_author':'CHONG LIU','render_dpi':int(os.environ.get('FIG_DPI','600'))},indent=2),encoding='utf-8')
