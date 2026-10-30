"""Compact transfer diagnostics, native-aspect workflow and sensitivity layouts."""
from f_common import *
import f02_fig2_transfer_bias as f2
import f08_supporting_figures as si


def _style(ax,xlabel=None,ylabel=None):
    style_axis(ax,xlabel=xlabel,ylabel=ylabel,tick_size=10.5,label_size=11.5)
    ax.tick_params(length=3.4,width=.8,pad=3)


def roc3(fig,ss,ctx):
    ax=fig.add_subplot(ss)
    ax.plot([0,1],[0,1],color=MUTED,lw=.8,ls=(0,(3,2)),zorder=1)
    for name,key,label in f2.MODELS:
        d=f2.ROC[f2.ROC.model==name]
        ax.plot(d.fpr,d.tpr,color=C[key],lw=1.65,
                label=f'{label} ({d.auc.iloc[0]:.2f})',zorder=3)
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    ax.legend(loc='lower right',handlelength=1.3,labelspacing=.3,
              borderaxespad=.35,fontsize=10.5)
    _style(ax,'False-positive rate','True-positive rate')


def class_bias3(fig,ss,ctx):
    g=ss.subgridspec(2,1,height_ratios=[.15,1],hspace=.035)
    hs=[Line2D([],[],ls='',marker=m,mfc=C[k],mec=INK,mew=.5,ms=5,label=l)
        for m,k,l in [('o','observed','Design estimate'),('D','plugin','Predicted')]]
    legend_strip(fig,g[0],hs,2,handlelength=1,handletextpad=.25,columnspacing=.7)
    ax=fig.add_subplot(g[1]);d=f2.CLS
    for y,r in enumerate(d.itertuples()):
        ax.plot([r.plugin*100,r.obs*100],[y,y],color=MUTED,lw=1)
        ax.errorbar(r.obs*100,y,xerr=[[(r.obs-r.lo)*100],[(r.hi-r.obs)*100]],
                    fmt='o',ms=5.8,mfc=C['observed'],mec=INK,mew=.5,
                    ecolor=INK,elinewidth=.8,capsize=2,zorder=3)
        ax.plot(r.plugin*100,y,ls='',marker='D',ms=5,mfc=C['plugin'],mec=INK,mew=.5)
    ax.set_yticks(range(len(d)),d.cls);ax.set_ylim(len(d)-.5,-.5)
    ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20])
    _style(ax,'Exceeding systems (%)','People served per system')


def transfer_curve3(fig,ss,ctx):
    f2.panel_c(fig,ss,ctx)
    # The four reading keys share one line in a full-width panel.
    axes=fig.axes[-2:];legend_ax,ax=axes
    lg=legend_ax.get_legend();handles=lg.legend_handles;labels=[t.get_text() for t in lg.get_texts()]
    lg.remove();legend_ax.legend(handles,labels,ncol=4,loc='center',
        handlelength=1.25,handletextpad=.25,columnspacing=.8)
    ax.axvspan(1.4,np.log10(3300),color=C['sampled'],alpha=.065,lw=0,zorder=0)
    ax.text(.02,.96,'Small systems',ha='left',va='top',transform=ax.transAxes,fontsize=10.5)
    ax.text(.58,.96,'Larger systems',ha='left',va='top',transform=ax.transAxes,fontsize=10.5)
    _style(ax,'People served per system','Exceeding systems (%)')


def layout2_round3(fig):
    g=fig.add_gridspec(2,2,height_ratios=[1,.92],left=.065,right=.99,top=.985,
                         bottom=.06,wspace=.14,hspace=.09)
    return [g[0,0],g[0,1],g[1,:]]


def workflow3(fig,ss,ctx):
    ax=fig.add_subplot(ss);ax.axis('off');ax.set_gid('schematic')
    asset=OUTPUT/'figures/Fig3_anchor_method/source/workflow_landscape.png'
    # Explicit extent follows the 3:1 asset at the intended physical panel size.
    if asset.exists():
        ax.imshow(plt.imread(asset),extent=[0,1,.17,.985],aspect='auto')
    else:
        # The illustrative raster is not redistributed with the source release.
        # A code-native scaffold preserves the workflow, without quantitative marks.
        from matplotlib.patches import Rectangle
        for x,c in zip([.02,.28,.55,.79],[C['plugin'],C['recal'],C['ratio'],C['people']]):
            ax.add_patch(Rectangle((x,.25),.18,.55,fc=c,ec=INK,lw=.7,alpha=.22))
    for x,label in [(.128,'Monitored larger\nsystems'),(.392,'Risk and error\nmodels'),
                    (.660,'Allocation +\nsampling depth'),(.894,'Systems +\npeople served')]:
        ax.text(x,.06,label,ha='center',va='center',fontsize=10.5,linespacing=1.1)
    for x0,x1 in [(.252,.271),(.535,.551),(.775,.793)]:
        ax.annotate('',xy=(x1,.55),xytext=(x0,.55),
                    arrowprops=dict(arrowstyle='-|>',color=INK,lw=.9))
    ax.set_xlim(0,1);ax.set_ylim(0,1)


def equation3(fig,ss,ctx):
    ax=fig.add_subplot(ss);ax.set_gid('schematic');ax.axis('off')
    terms=[r'$\lambda\sum_U a_i f_i$',
           r'$\sum_s\frac{a_i(\widetilde y_i-\lambda f_i)}{\pi_i}$',
           r'$\sum_{s_2}\frac{a_i(y_i-\widetilde y_i)}{\pi_i q}$']
    ax.text(.002,.57,r'$\widehat\theta=\frac{1}{A}\,[\,$',fontsize=15.5,ha='left',va='center')
    for x,t,c,label in zip([.27,.54,.83],terms,[C['plugin'],C['recal'],C['ratio']],
                         ['Model total','Weighted residuals','Random-completion correction']):
        ax.text(x,.60,t,fontsize=13.5,ha='center',va='center')
        ax.plot([x-.105,x+.105],[.24,.24],color=c,lw=2)
        ax.text(x,.07,label,fontsize=10.5,ha='center',va='center')
    for x in [.397,.688]:ax.text(x,.57,'+',ha='center',va='center',fontsize=16)
    ax.text(.99,.57,']',ha='right',va='center',fontsize=18)
    ax.set_xlim(0,1);ax.set_ylim(0,1)


def layout3_round3(fig):
    g=fig.add_gridspec(3,1,height_ratios=[1.25,.40,1.12],left=.055,right=.99,
                         top=.99,bottom=.05,hspace=.045)
    low=g[2].subgridspec(1,2,width_ratios=[1,1.05],wspace=.16)
    return [g[0],g[1],low[0],low[1]]


def fraction3(eff=False):
    builder=si.fraction_panel(eff)
    def draw(fig,ss,ctx):
        builder(fig,ss,ctx);ax=fig.axes[-1]
        ax.set_xlabel(r'Full-schedule fraction $q$')
        if eff:ax.set_ylabel('Efficiency vs full ANCHOR')
        _style(ax)
    return draw


def supplement1_round3(fig):
    g=fig.add_gridspec(4,2,height_ratios=[.11,1,.12,1],left=.065,right=.99,
                         top=.99,bottom=.055,wspace=.14,hspace=.045)
    legend_strip(fig,g[0,:],[Line2D([],[],color=c,marker='o',ms=4,label=l)
                            for _,l,c in si.IV],4,handlelength=1.3)
    legend_strip(fig,g[2,:],[Line2D([],[],color=c,marker='o',ms=4,label=str(n))
                            for c,n in zip([PAL[1],PAL[3],PAL[4]],[150,300,600])],
                 3,title='Budget (full-schedule system equivalents)',labelspacing=.15)
    return [g[1,0],g[1,1],g[3,0],g[3,1]]


def supplement2_round3(fig):
    from matplotlib.colors import Normalize
    heat=LinearSegmentedColormap.from_list('white_teal',['white',PAL[3]])
    g=fig.add_gridspec(5,2,height_ratios=[1.04,.20,.11,.67,1.12],left=.065,right=.99,
                         top=.99,bottom=.045,wspace=.14,hspace=.045)
    cax=fig.add_subplot(g[1,:]);p=cax.get_position()
    cax.set_position([p.x0+.20*p.width,p.y0+.72*p.height,p.width*.60,p.height*.22])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,5),cmap=heat),cax=cax,orientation='horizontal')
    cb.set_ticks([0,1,3,5]);cb.set_label('Efficiency vs reconstructed cells',fontsize=10.5,labelpad=2)
    hs=[Line2D([],[],ls='',marker=m,color=c,mec=INK,mew=.5,label=l,ms=5)
        for c,m,l in [(PAL[1],'s','Geography'),(PAL[3],'D','+ Neighbours')]]
    legend_strip(fig,g[2,:],hs,2)
    return [g[0,0],g[0,1],g[3,0],g[3,1],g[4,:]]
