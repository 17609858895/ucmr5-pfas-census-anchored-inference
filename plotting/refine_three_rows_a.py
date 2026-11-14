"""Three-row main-figure compositions for Figs 1–4.

All observations, grouping rules and numeric summaries are retained. Fig. 3's
equation is deliberately absent: its native mathematical form belongs in Methods.
The official main entry point supplies the same independent panel export helper.
"""
import os
from f_common import *
import figstyle as fs
import f01_fig1_monitoring_gap as f1
import f03_fig3_anchor_method as f3
import f04_fig4_census_validation as f4
import refine_overview_round3 as ov3
import refine_mechanism_round3 as rm3


def _style(ax, xlabel=None, ylabel=None):
    style_axis(ax, xlabel=xlabel, ylabel=ylabel, tick_size=10.5, label_size=11.5)
    ax.tick_params(length=3.4, width=.8, pad=3)


FIG1_CANVAS_MM = (190,187)
FIG1_AB_EXTRA_MM = 3.0


def layout1_three_rows(fig):
    """Keep original physical panel sizes; add 3 mm only between a and b."""
    old_height=184.0
    usable=old_height*(.985-.035)
    gap=usable*.040/(3+2*.040)
    unit=(usable-2*gap)/(1.85+.80+.75)
    height_mm=fig.get_figheight()*25.4
    g=fig.add_gridspec(5,1,
        height_ratios=[1.85*unit,gap+FIG1_AB_EXTRA_MM,.80*unit,gap,.75*unit],
        left=.065,right=.99,top=1-old_height*.015/height_mm,
        bottom=old_height*.035/height_mm,hspace=0)
    return [g[0],g[2],g[4]]


# Fig. 1 fourth row (2026-11-15): the eight working post-stratification cells. Rows a–c keep their
# physical heights; the canvas grows by the new row so no existing panel is rescaled.
FIG1_ROW4_MM = 31.0
FIG1_ROW4_GAP_MM = 6.0
FIG1_FOUR_ROW_CANVAS_MM = (190, round(FIG1_CANVAS_MM[1]+FIG1_ROW4_MM+FIG1_ROW4_GAP_MM,1))
CELL_SOURCES = [('GW','GW'),('SW','SW')]
CELL_SIZES = ['<=100','101-500','501-1000','1001-3300']


def layout1_four_rows(fig):
    old_height=184.0
    usable=old_height*(.985-.035)
    gap=usable*.040/(3+2*.040)
    unit=(usable-2*gap)/(1.85+.80+.75)
    height_mm=fig.get_figheight()*25.4
    g=fig.add_gridspec(7,1,
        height_ratios=[1.85*unit,gap+FIG1_AB_EXTRA_MM,.80*unit,gap,.75*unit,FIG1_ROW4_GAP_MM,FIG1_ROW4_MM],
        left=.065,right=.99,top=1-old_height*.015/height_mm,
        bottom=old_height*.035/height_mm,hspace=0)
    return [g[0],g[2],g[4],g[6]]


def cell_exceedance4(fig,ss,ctx):
    """Fig. 1d: unweighted observed exceedance among monitored systems in the eight working cells.
    Shading and cell text give exceeding/monitored systems; Table S3 lists each cell's monitored
    share. No weighting or modelling."""
    from matplotlib.colors import LinearSegmentedColormap
    t=pd.read_csv(FD/'fig1_cell_summary.csv')
    g=ss.subgridspec(1,2,width_ratios=[1,.022],wspace=.03)
    ax=fig.add_subplot(g[0]);cax=fig.add_subplot(g[1])
    cmap=LinearSegmentedColormap.from_list('cell_exceedance',['white',C['systems']])
    cell=lambda s,z:t[(t.source==s)&(t.size_band==z)].iloc[0]
    m=np.array([[float(cell(s,z).observed_exceedance_percent) for z in CELL_SIZES] for s,_ in CELL_SOURCES])
    im=ax.imshow(m,cmap=cmap,vmin=0,vmax=22,aspect='auto')
    for i,(s,_) in enumerate(CELL_SOURCES):
        for j,z in enumerate(CELL_SIZES):
            r=cell(s,z)
            ax.text(j,i,f'{r.observed_exceedance_percent:.1f}% ({int(r.observed_exceeding)}/{int(r.observed_systems)})',
                    ha='center',va='center',fontsize=11,color=INK)
    ax.set_xticks(range(4),CLASS_ORDER);ax.set_yticks(range(2),['Groundwater','Surface water'])
    ax.set_xticks(np.arange(-.5,4),minor=True);ax.set_yticks(np.arange(-.5,2),minor=True)
    ax.grid(which='minor',color='white',lw=1.6);ax.tick_params(which='minor',length=0)
    ax.tick_params(length=0,labelsize=10.5,pad=3)
    for sp in ax.spines.values():sp.set_linewidth(.8)
    ax.set_xlabel('System size (people)',fontsize=11.5,labelpad=3)
    cb=fig.colorbar(im,cax=cax);cb.outline.set_linewidth(.6);cb.set_ticks([0,10,20])
    cb.ax.tick_params(labelsize=10,length=2.5,width=.6,pad=2);cb.set_label('Exceeding (%)',fontsize=10.5,labelpad=3)


def small_composition4(fig,ss,ctx):
    """Fig. 1c, with a one-line key suited to the full-width third row."""
    hs=[Patch(fc=C[f'cls{i}'],ec=INK,lw=.5,label=cls)
        for i,cls in enumerate(CLASS_ORDER)]
    if ctx['standalone']:
        # Keep the existing standalone exactly; the reallocation only concerns
        # the combined Fig. 1 map/composition balance.
        g=ss.subgridspec(2,1,height_ratios=[.60,1],hspace=.035)
        legend_strip(fig,g[0],hs,4,title='System size (people)',
                     handlelength=1.0,columnspacing=1.1,labelspacing=.15)
        ax=fig.add_subplot(g[1])
    else:
        # A right-hand key frees vertical room for the unchanged bar numbers.
        g=ss.subgridspec(1,2,width_ratios=[3.2,1.15],wspace=.07)
        legend_strip(fig,g[1],hs,1,title='System size (people)',
                     handlelength=1.0,labelspacing=.18)
        ax=fig.add_subplot(g[0])
    rows=['Frame: systems','Frame: people','UCMR5 sample: systems']
    left=np.zeros(3)
    for i,cls in enumerate(CLASS_ORDER):
        values=f1.COMP.loc[cls,rows].to_numpy()*100
        ax.barh(range(3),values,left=left,height=.78,color=C[f'cls{i}'],
                edgecolor=INK,lw=.5,zorder=2)
        for j,(start,value) in enumerate(zip(left,values)):
            if value>=10:ax.text(start+value/2,j,f'{value:.0f}',
                                ha='center',va='center',fontsize=10.5)
        left+=values
    ax.set_yticks(range(3),['Systems','People','UCMR5 sample'])
    ax.set_ylim(2.55,-.55);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
    _style(ax,'Share within small systems (%)')


def layout2_three_rows(fig):
    """ROC / size-stratified bias / scale-transfer curve."""
    g=fig.add_gridspec(3,1,height_ratios=[1.05,.9,1.0],
        left=.065,right=.99,top=.985,bottom=.045,hspace=.08)
    return [g[0],g[1],g[2]]


def layout3_three_rows(fig):
    """Workflow / inclusion probability / expected composition; no equation."""
    g=fig.add_gridspec(3,1,height_ratios=[1.30,1.0,.94],
        left=.06,right=.99,top=.99,bottom=.035,hspace=.038)
    return [g[0],g[1],g[2]]


def allocation4(fig,ss,ctx):
    """Original Fig. 3c -> new Fig. 3b; all medians/quantile bands unchanged."""
    g=ss.subgridspec(2,1,height_ratios=[.18,1],hspace=.035)
    designs=[('pi_epa','Cell reference',C['PPS-pop + Hájek (EPA-like)'],'s'),
             ('pi_anchor','ANCHOR (systems)',C['ANCHOR (Poisson + PPI++)'],'D'),
             ('pi_dual','Dual design',C['Neyman-H10 + PPI++ (hybrid)'],'o')]
    hs=[Line2D([],[],color=color,lw=1.4,marker=marker,ms=4.5,mec=INK,mew=.45,label=label)
        for _,label,color,marker in designs]
    legend_strip(fig,g[0],hs,3,handlelength=1.3,labelspacing=.2,columnspacing=1.05)
    ax=fig.add_subplot(g[1]);d=f3.PI
    bins=np.quantile(np.log10(d.f),np.linspace(0,1,11))
    group=np.clip(np.digitize(np.log10(d.f),bins[1:-1]),0,9)
    x=d.groupby(group).f.median().values
    for key,label,color,marker in designs:
        med=d.groupby(group)[key].median().values
        lo=d.groupby(group)[key].quantile(.1).values
        hi=d.groupby(group)[key].quantile(.9).values
        ax.fill_between(x,lo*100,hi*100,color=color,alpha=.2,lw=0,zorder=1)
        ax.plot(x,med*100,color=color,lw=1.5,marker=marker,ms=4.5,mec=INK,mew=.45,zorder=3)
    ax.set_xscale('log');ax.set_yscale('log');ax.minorticks_off()
    ax.set_xticks([.003,.01,.03,.1,.3],['0.3','1','3','10','30'])
    ax.set_yticks([.3,1,3,10],['0.3','1','3','10']);ax.set_ylim(.2,15)
    _style(ax,'Predicted risk (%)','Inclusion probability (%)')


def expected_composition4(fig,ss,ctx):
    """Original Fig. 3d -> new Fig. 3c; frame-weighted sum of original pi."""
    g=ss.subgridspec(2,1,height_ratios=[.44,1],hspace=.035)
    hs=[Patch(fc=C[f'cls{i}'],ec=INK,lw=.45,label=cls)
        for i,cls in enumerate(CLASS_ORDER)]
    legend_strip(fig,g[0],hs,4,title='System size (people)',
                 handlelength=1.0,columnspacing=1.1,labelspacing=.15)
    ax=fig.add_subplot(g[1]);d=f3.PI
    classes=['<=100','101-500','501-1000','1001-3300']
    arr=np.array([d.groupby('scls')[key].sum().reindex(classes).values/d[key].sum()*100
                  for key in ['pi_epa','pi_anchor','pi_dual']])
    left=np.zeros(3)
    for i in range(4):
        values=arr[:,i]
        ax.barh(range(3),values,left=left,height=.76,color=C[f'cls{i}'],
                edgecolor=INK,lw=.45,zorder=2)
        for j,(start,value) in enumerate(zip(left,values)):
            if value>=12:ax.text(start+value/2,j,f'{value:.0f}',
                                ha='center',va='center',fontsize=10.5)
        left+=values
    ax.set_yticks(range(3),['Cell reference','ANCHOR','Dual'])
    ax.set_ylim(2.5,-.5);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
    _style(ax,'Expected sample composition (%)')


def layout4_three_rows(fig):
    """Six unchanged quantitative panels in a compact three-by-two grid."""
    g=fig.add_gridspec(4,1,height_ratios=[.22,.96,1.0,1.10],
        left=.065,right=.99,top=.99,bottom=.035,hspace=.028)
    hs=f4.method_handles()
    for h in hs:
        if h.get_label()=='ANCHOR':h.set_markerfacecolor('none');h.set_markersize(6.3)
    legend_strip(fig,g[0],hs,4,handlelength=1.55,columnspacing=.75,labelspacing=.25)
    first=g[1].subgridspec(1,2,wspace=.09)
    second=g[2].subgridspec(1,2,wspace=.09)
    third=g[3].subgridspec(1,2,width_ratios=[1,1.12],wspace=.09)
    return [first[0],first[1],second[0],second[1],third[0],third[1]]


def pareto4(fig,ss,ctx):
    """Keep the SRS point label clear of the lower axis in the third row."""
    f4.panel_pareto(fig,ss,ctx)
    for text in fig.axes[-1].texts:
        if text.get_text()=='SRS':
            text.set_position((6,5));text.set_verticalalignment('bottom')


SIZES={'1':FIG1_CANVAS_MM,'2':(190,194),'3':(190,194),'4':(190,184)}


def preview():
    """Drafts use final alignment and audit helpers but write only internally."""
    import generate_refined_figures as base
    from publication_layout import draw_module
    fs.MIN_PT=10.5;fs.LETTER_PT=12
    os.environ['FIG_DPI']='200'
    out=OUTPUT/'checks/draft_three_rows';out.mkdir(parents=True,exist_ok=True)
    fs.QA=out/'qa'
    plans=[
      ('Fig1_three_rows',[ov3.coverage_map3,ov3.coverage_by_size3,small_composition4],layout1_three_rows,SIZES['1']),
      ('Fig2_three_rows',[rm3.roc3,rm3.class_bias3,rm3.transfer_curve3],layout2_three_rows,SIZES['2']),
      ('Fig3_three_rows',[rm3.workflow3,allocation4,expected_composition4],layout3_three_rows,SIZES['3']),
      ('Fig4_three_rows',[ov3.efficiency_round3('medium','systems',ylim=(0,2.65),yticks=[0,1,2]),
          ov3.efficiency_round3('large','systems',ylim=(0,2.35),yticks=[0,1,2]),
          ov3.efficiency_round3('medium','people',ylim=(0,3.15),yticks=[0,1,2,3]),
          ov3.efficiency_round3('large','people',logy=True,ylim=(.3,70),yticks=[.3,1,3,10,30]),
          pareto4,ov3.coverage_round3],layout4_three_rows,SIZES['4'])]
    for stem,builders,layout,size in plans:
        fig=plt.figure(figsize=np.array(size)/25.4)
        for builder,cell,letter in zip(builders,layout(fig),'abcdef'):
            draw_module(fig,cell,letter,base.polish(builder))
        report=fs._export(fig,out,stem,True)
        print(stem,report['alignment_max_residual_pt'],report['n_unexpected'],len(report['label_overlaps']),flush=True)

if __name__=='__main__':preview()
