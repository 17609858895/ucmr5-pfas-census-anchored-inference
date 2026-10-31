from strengthening import paths
"""Three-row Figure2 and FigureS4; plotting reads frozen analysis CSVs only."""
from pathlib import Path
import sys,os,json,shutil,hashlib,argparse
P=paths.SPATIAL_RESULTS;R=paths.WORKDIR
OLD=paths.PLOTTING_CODE
WORK=P/'图件';WORK.mkdir(exist_ok=True);(WORK/'figure_data').mkdir(exist_ok=True)
for name in ['fig2_roc.csv','fig2_size_curve.csv','fig2_by_class.csv']:
    src=paths.WORKDIR/'figure_data'/name;dst=WORK/'figure_data'/name
    if not dst.exists():shutil.copy2(src,dst)
    assert src.read_bytes()==dst.read_bytes()
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'figkit'))
import figstyle as fs
import publication_layout as layout
fs.OUTPUT=WORK;fs.FIG=WORK/'figures';fs.QA=WORK/'figures/qa';fs.RES=OLD/'results'
layout.OUTPUT=WORK;layout.FIG=WORK/'figures'
from f_common import *
import f02_fig2_transfer_bias as original
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import Text
from audit_palette_usage import validate_series_colors
fs.MIN_PT=12;fs.LETTER_PT=12
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':12,
                    'axes.labelsize':13.5,'axes.labelweight':'normal','font.weight':'normal',
                    'axes.titleweight':'normal','xtick.labelsize':12.5,'ytick.labelsize':12.5,
                    'legend.fontsize':12,'legend.title_fontsize':12,'axes.linewidth':.9,'lines.linewidth':1.6,'savefig.dpi':600})
COL={'available':PAL[1],'cold':PAL[4],'attributes':C['m_sdwis'],'geography':C['m_geo'],'neighbours':C['m_nb'],'systems':C['systems'],'people':C['people']}
validate_series_colors('nature_new',COL)
M=pd.read_csv(P/'information_regime_metrics.csv');M=M[M.weighting=='systems']
B=pd.read_csv(P/'burden_model_stress.csv')
CURVES=pd.read_csv(P/'plot_roc_curves.csv')
BUFFER=pd.read_csv(P/'buffer_matched_metrics.csv')

# Record actual artists after the shared layout pass; no global helper is edited.
_base_audit_style=fs.audit_style
def audit_regular_spatial_text(fig):
    report=_base_audit_style(fig)
    actual=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    report['actual_text_styles']=[dict(text=t.get_text(),weight=t.get_fontweight(),
        size_pt=t.get_fontsize(),family=t.get_fontfamily()) for t in actual]
    report['all_actual_text_normal']=all(t.get_fontweight() in ('normal',400) for t in actual)
    report['bold_math_commands']=[t.get_text() for t in actual
        if any(cmd in t.get_text() for cmd in [r'\mathbf',r'\boldsymbol',r'\mathbfit'])]
    return report
fs.audit_style=audit_regular_spatial_text

def sty(ax,x=None,y=None):
    style_axis(ax,xlabel=x,ylabel=y,tick_size=12.5,label_size=13.5)
    ax.tick_params(length=3.5,width=.9)
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_weight('normal')

def information_handles():
    return [Line2D([],[],ls='',marker='o',mfc=COL['available'],mec=INK,mew=.6,ms=5.5,label='Available layer'),
            Line2D([],[],ls='',marker='D',mfc='white',mec=COL['cold'],mew=1.3,ms=5.5,label='Region excluded')]

def auc_forest(models,labels):
    def draw(fig,ss,ctx):
        g=ss.subgridspec(2,1,height_ratios=[.22,1],hspace=.10)
        legend_strip(fig,g[0],information_handles(),2,handlelength=.7,columnspacing=1.4)
        ax=fig.add_subplot(g[1])
        for i,model in enumerate(models):
            vals=[]
            for regime,off,mk in [('available',-.13,'o'),('cold',.13,'D')]:
                r=M[(M.regime==regime)&(M.model==model)].iloc[0];vals.append(r.auc)
                ax.errorbar(r.auc,i+off,xerr=[[r.auc-r.auc_lo],[r.auc_hi-r.auc]],fmt=mk,ms=6,
                            mfc=COL[regime] if regime=='available' else 'white',mec=INK if regime=='available' else COL[regime],mew=.7 if regime=='available' else 1.2,
                            ecolor=COL[regime],elinewidth=1.2,capsize=2,zorder=3)
            ax.plot(vals,[i-.13,i+.13],color=MUTED,lw=.7,zorder=1)
        ax.axvline(.5,color=MUTED,ls='--',lw=.8,zorder=0)
        ax.set_yticks(range(len(models)),labels);ax.set_ylim(len(models)-.5,-.5)
        ax.set_xlim(.2,1);ax.set_xticks([.2,.4,.6,.8,1]);sty(ax,'AUC')
    return draw

def burden(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.19,1],hspace=.08)
    raw=tuple(Line2D([],[],ls='',marker='s',ms=5.3,mfc='white',mec=COL[e],mew=1.2) for e in ['systems','people'])
    corr=tuple(Line2D([],[],ls='',marker='o',ms=5.3,mfc=COL[e],mec=INK,mew=.5) for e in ['systems','people'])
    la=fig.add_subplot(g[0]);la.axis('off');la.legend([raw,corr],['Prediction mean','Working correction'],handler_map={tuple:HandlerTuple(ndivide=None,pad=.15)},loc='center',ncol=2,handlelength=2,handletextpad=.35,columnspacing=1.0)
    grid=g[1].subgridspec(1,2,wspace=.15)
    keys=['frozen_production','available_neighbours','cold_neighbours','cold_coordinates_neighbours']
    labels=['Production\nmodel','Available\nneighbours','Region-excluded\nneighbours','Region-excluded\nno IDs']
    for j,(target,xlabel) in enumerate([('systems','Exceeding systems (%)'),('people','Associated service\npopulation (%)')]):
        ax=fig.add_subplot(grid[j]);ref=B[(B.estimand==target)&(B.model=='direct_poststratified')].theta.iloc[0]
        ax.axvline(ref*100,color=MUTED,lw=.8,ls='--',zorder=0)
        for y,key in enumerate(keys):
            r=B[(B.estimand==target)&(B.model==key)].iloc[0]
            ax.plot([r.prediction_mean*100,r.theta*100],[y,y],color=MUTED,lw=.7,zorder=1)
            ax.errorbar(r.theta*100,y,xerr=[[(r.theta-r.kg_lo)*100],[(r.kg_hi-r.theta)*100]],fmt='o',ms=5.7,mfc=COL[target],mec=INK,mew=.5,ecolor=INK,elinewidth=.9,capsize=2,zorder=3)
            ax.plot(r.prediction_mean*100,y,ls='',marker='s',ms=5.4,mfc='white',mec=COL[target],mew=1.2,zorder=3)
        ax.set_yticks(range(4),labels if j==0 else ['']*4);ax.set_ylim(3.6,-.6);ax.set_xlim(0,12);ax.set_xticks([0,4,8,12]);sty(ax,xlabel)

def production_size(fig,ss,ctx):
    prior=original.STRIP;original.STRIP=.18
    original.panel_c(fig,ss,ctx);original.STRIP=prior
    legend_ax,ax=fig.axes[-2:];lg=legend_ax.get_legend();hs=lg.legend_handles;lg.remove()
    legend_ax.legend(hs,['Larger obs.','Larger pred.','Small obs.','Small pred.'],ncol=4,loc='center',handlelength=1.1,handletextpad=.25,columnspacing=.6)
    ax.axvspan(1.4,np.log10(3300),color=C['sampled'],alpha=.065,lw=0,zorder=0)
    sty(ax,'System size (people)','Exceeding systems (%)')

def layout_main(fig):
    g=fig.add_gridspec(3,1,height_ratios=[.93,1.12,1],left=.065,right=.99,top=.99,bottom=.045,hspace=.14)
    return [g[i] for i in range(3)]

def roc(regime):
    def draw(fig,ss,ctx):
        from sklearn.metrics import roc_curve
        if ctx['standalone']:
            g=ss.subgridspec(2,1,height_ratios=[.29,1],hspace=.10)
            legend_strip(fig,g[0],[Line2D([],[],color=COL[k],lw=1.6,label=l) for k,l in [('attributes','Attributes'),('geography','+ Geography'),('neighbours','+ Neighbours')]],1,labelspacing=.2)
            ax=fig.add_subplot(g[1])
        else:ax=fig.add_subplot(ss)
        ax.plot([0,1],[0,1],color=MUTED,ls='--',lw=.8,zorder=0)
        for name,label in [('attributes','Attributes'),('geography','+ Geography'),('neighbours','+ Neighbours')]:
            key=name if regime=='available' else 'cold_'+name
            curve=CURVES[CURVES.regime.eq(regime)&CURVES.model.eq(name)];x=curve.fpr.to_numpy();y=curve.tpr.to_numpy();point=M[(M.regime==regime)&(M.model==name)].auc.iloc[0]
            ax.plot(x,y,color=COL[name],lw=1.6,label=f'{label} ({point:.2f})')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
        sty(ax,'False-positive rate','True-positive rate')
    return draw

def buffer_auc(fig,ss,ctx):
    ax=fig.add_subplot(ss)
    for y,(key,label,color,mk) in enumerate([('unbuffered_geo_complete_source','0 km',PAL[1],'o'),('buffer50','50 km',PAL[4],'D')]):
        r=BUFFER[BUFFER.regime==key].iloc[0]
        ax.errorbar(r.auc,y,xerr=[[r.auc-r.auc_lo],[r.auc_hi-r.auc]],fmt=mk,ms=6,mfc=color,mec=INK,mew=.5,ecolor=color,elinewidth=1.2,capsize=2)
    ax.axvline(.5,color=MUTED,ls='--',lw=.8);ax.set_yticks([0,1],['0 km','50 km']);ax.set_ylim(1.6,-.6);ax.set_xlim(.2,.8);ax.set_xticks([.2,.4,.6,.8]);sty(ax,'AUC','Spatial buffer')

def no_id_auc(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.20,1],hspace=.10)
    handles=information_handles()
    handles[0].set_label('Available');handles[1].set_label('Excluded')
    legend_strip(fig,g[0],handles,2,handlelength=.7,columnspacing=.7)
    ax=fig.add_subplot(g[1])
    for x,model in enumerate(['coordinates','coordinates_neighbours']):
        for regime,off,mk in [('available',-.10,'o'),('cold',.10,'D')]:
            r=M[(M.regime==regime)&(M.model==model)].iloc[0]
            ax.errorbar(x+off,r.auc,yerr=[[r.auc-r.auc_lo],[r.auc_hi-r.auc]],fmt=mk,ms=6,
                        mfc=COL[regime] if regime=='available' else 'white',mec=INK if regime=='available' else COL[regime],mew=.6 if regime=='available' else 1.2,
                        ecolor=COL[regime],elinewidth=1.2,capsize=2)
    ax.axhline(.5,color=MUTED,ls='--',lw=.8);ax.set_xlim(-.45,1.45);ax.set_ylim(.2,1)
    ax.set_xticks([0,1],['Coordinates','+ Neighbours']);ax.set_yticks([.2,.4,.6,.8,1]);sty(ax,y='AUC')

def small_size(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.25,1],hspace=.10)
    legend_strip(fig,g[0],[Line2D([],[],ls='',marker='o',ms=5.3,mfc=C['observed'],mec=INK,mew=.5,label='Observed'),Line2D([],[],ls='',marker='D',ms=5.3,mfc=C['plugin'],mec=INK,mew=.5,label='Predicted')],2,handlelength=.8,columnspacing=.7)
    ax=fig.add_subplot(g[1]);d=original.CLS
    for y,r in enumerate(d.itertuples()):
        ax.plot([r.plugin*100,r.obs*100],[y,y],color=MUTED,lw=.7)
        ax.errorbar(r.obs*100,y,xerr=[[(r.obs-r.lo)*100],[(r.hi-r.obs)*100]],fmt='o',ms=5.8,mfc=C['observed'],mec=INK,mew=.5,ecolor=INK,elinewidth=.8,capsize=2)
        ax.plot(r.plugin*100,y,ls='',marker='D',ms=5.5,mfc=C['plugin'],mec=INK,mew=.5)
    ax.set_yticks(range(4),d.cls);ax.set_ylim(3.5,-.5);ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20]);sty(ax,'Exceeding systems (%)','System size (people)')

def regional_support(fig,ss,ctx):
    d=pd.read_csv(P/'region_holdout/spatial_region_metrics.csv');d=d[(d.cohort=='small_observed')&(d.model=='neighbours')]
    s=pd.read_csv(P/'neighbour_availability_summary.csv');s=s[s.region!='all'].copy();s['region']=s.region.astype(int)
    d=d.merge(s[['region','cold_fraction']],on='region').dropna(subset=['auc'])
    ax=fig.add_subplot(ss);ax.scatter(d.cold_fraction,d.auc,s=38,color=COL['cold'],edgecolor=INK,linewidth=.5,zorder=3)
    offsets={1:(7,4),2:(5,4),3:(6,-5),4:(-6,8),5:(6,7),6:(6,5),7:(6,-4),9:(8,-18),10:(5,4)}
    for r in d.itertuples():
        dx,dy=offsets[r.region]
        kw={'bbox':dict(facecolor='white',edgecolor='none',pad=.1),'arrowprops':dict(arrowstyle='-',color=MUTED,lw=.6,shrinkA=2,shrinkB=3)} if r.region==9 else {}
        ax.annotate('R'+str(r.region),(r.cold_fraction,r.auc),xytext=(dx,dy),textcoords='offset points',fontsize=12,ha='left' if dx>=0 else 'right',va='bottom',**kw)
    ax.axhline(.5,color=MUTED,ls='--',lw=.8,zorder=0)
    ax.set_xlim(-.025,.55);ax.set_ylim(.05,1);ax.set_xticks([0,.25,.5]);ax.set_yticks([0,.25,.5,.75,1]);sty(ax,'Neighbour support\nfraction','Regional AUC')

def layout_si(fig):
    g=fig.add_gridspec(4,2,height_ratios=[.16,1,1.05,1.12],left=.065,right=.99,top=.99,bottom=.045,wspace=.20,hspace=.16)
    legend_strip(fig,g[0,:],[Line2D([],[],color=COL[k],lw=1.6,label=l) for k,l in [('attributes','Attributes'),('geography','+ Geography'),('neighbours','+ Neighbours')]],3,handlelength=1.1,columnspacing=1.1)
    return [g[1,0],g[1,1],g[2,0],g[2,1],g[3,0],g[3,1]]

def run(draft=False,only=None):
    if draft:
        layout.FIG=WORK/'draft';fs.QA=WORK/'draft/qa';os.environ['FIG_DPI']='180';os.environ['FIG_COMBINED_ONLY']='1'
    else:
        os.environ.pop('FIG_COMBINED_ONLY',None);os.environ['FIG_DPI']='600'
    tasks={
      '2':([auc_forest(['attributes','geography','neighbours'],['Attributes','+ Geography','+ Neighbours']),burden,production_size],layout_main,(190,218),[(185,88),(190,115),(185,85)],'Fig_2'),
      'S4':([roc('available'),roc('cold'),no_id_auc,buffer_auc,small_size,regional_support],layout_si,(190,238),[(110,105),(110,105),(125,100),(110,95),(125,105),(115,100)],'Fig_S4')}
    for n in only or tasks:
        builders,plan,size,subsizes,folder=tasks[n]
        layout.save_modules(builders,folder,folder,plan,figsize_mm=size,sizes_mm=subsizes)
    if not draft:
        dest=paths.FIGURES
        for n in only or tasks:
            folder=tasks[n][-1];src=WORK/'figures'/folder;to=dest/folder
            shutil.copytree(src,to,dirs_exist_ok=True)
        (WORK/'source_input_sha256.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'information_regime_metrics.csv',P/'burden_model_stress.csv',P/'buffer_matched_metrics.csv',P/'plot_roc_curves.csv',paths.WORKDIR/'figure_data/fig2_size_curve.csv',paths.WORKDIR/'figure_data/fig2_by_class.csv']},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--draft',action='store_true');a.add_argument('--only',nargs='*');x=a.parse_args();run(x.draft,x.only)
