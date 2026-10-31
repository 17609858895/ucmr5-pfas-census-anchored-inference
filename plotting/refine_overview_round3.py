"""Third-round overview composition; all observations and results are frozen.

CONUS has a larger geographic viewport; Alaska and Hawaii retain their original
projected coordinates in explicitly labelled viewports.  Numeric comparisons use
the existing f04 builders and retain all seven design/estimator series.
"""
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle
from f_common import *
import figstyle as fs
import f01_fig1_monitoring_gap as f1
import f04_fig4_census_validation as f4


def _region_mask(lon, lat, region):
    ak=(lat > 50) & (lon < -129)
    hi=(lon < -150) & (lat < 25) & (lat > 15)
    return ak if region == 'AK' else hi if region == 'HI' else ~(ak | hi)


def _land(ax, region='CONUS'):
    rings=[]
    for st, poly in fs.load_states():
        if (region == 'CONUS' and st in ['AK','HI']) or (region != 'CONUS' and st != region):
            continue
        for p in poly:
            ax.fill(p[:,0],p[:,1],facecolor=fs.LAND,edgecolor=fs.BORDER,lw=.32,zorder=1)
            rings.append(p)
    ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    return np.vstack(rings)


def _status(ax, region='CONUS', zoom=False):
    bounds=_land(ax,region)
    for nm,key,_,_,_ in f1.CATS:
        d=f1.P[f1.P.category == nm].dropna(subset=['lat','lon'])
        keep=_region_mask(d.lon.to_numpy(),d.lat.to_numpy(),region)
        d=d[keep]
        map_points(ax,d.lon.to_numpy(),d.lat.to_numpy(),
          s=(7 if zoom else 4.8) if key == 'sampled' else (2.3 if zoom else .9),
          color=C[key],edgecolor=INK if key == 'sampled' else 'none',
          linewidths=.32 if key == 'sampled' else 0,zorder=4 if key == 'sampled' else 2)
    return bounds


FIG1_MAP_STANDALONE_MM = (195,105)


def coverage_map3(fig,ss,ctx):
    """All geographic points retained, with explicit AK/HI and regional viewports."""
    # The equal-aspect CONUS viewport needs real height to fill its column.
    # Fig. 1's own three-row layout supplies that height; the zooms share its
    # full map row, with a narrow physical gap rather than a vacant map column.
    g=ss.subgridspec(2,2,height_ratios=[1,.09],width_ratios=[3.1,1],wspace=.03,hspace=.055)
    ax=fig.add_subplot(g[0,0]);xy=_status(ax)
    ax.set_xlim(xy[:,0].min()-.015,max(xy[:,0].max()+.015,.415))
    ax.set_ylim(.018,xy[:,1].max()+.01)
    ax.set_gid('physical_basemap')
    ax.set_anchor('W')
    # Preserve existing Albers/inset coordinates; these are explicit viewports.
    for reg,box in [('AK',[.005,.008,.23,.22]),('HI',[.25,.008,.15,.13])]:
        ia=ax.inset_axes(box);points=_status(ia,reg)
        lo=points.min(axis=0);hi=points.max(axis=0);span=hi-lo
        ia.set_xlim(lo[0]-.04*span[0],hi[0]+.04*span[0])
        ia.set_ylim(lo[1]-.04*span[1],hi[1]+.04*span[1])
        ia.text(.02,1.03,reg,transform=ia.transAxes,fontsize=10.5,ha='left',va='bottom')
    zoom=g[0,1].subgridspec(3,1,hspace=.25)
    for j,name in enumerate(['Northeast corridor','Carolinas–Georgia','Southern California']):
        z=fig.add_subplot(zoom[j]);_status(z,zoom=True);z.set_anchor('W')
        x0,x1,y0,y1=fs.zoom_bbox(name);xm=(x0+x1)/2;ym=(y0+y1)/2
        width=max(x1-x0,(y1-y0)*2);height=width/2
        z.set_xlim(xm-width/2,xm+width/2);z.set_ylim(ym-height/2,ym+height/2)
        short={'Northeast corridor':'NE corridor','Carolinas–Georgia':'Carolinas–GA','Southern California':'S. California'}[name]
        z.set_title(short,loc='left',fontsize=12,pad=2)
        for sp in z.spines.values():sp.set_visible(True);sp.set_color(MUTED);sp.set_linewidth(.6)
        ax.add_patch(Rectangle((xm-width/2,ym-height/2),width,height,fill=False,ec=INK,lw=.55,zorder=8))
    n=f1.P.category.value_counts()
    hs=[Line2D([],[],ls='',marker='o',mfc=C[k],mec=INK,mew=.35,ms=4.6,
               label=f'{lab}\nn = {n[nm]:,}') for nm,k,lab in [
      ('Unmonitored','unmonitored','Unmonitored'),('Census (≥3,300)','census','Larger systems'),
      ('Sampled small (<3,300)','sampled','Monitored small')]]
    legend_strip(fig,g[1,:],hs,3,columnspacing=1.5,handletextpad=.3,handlelength=.6)


def layout1_round3(fig):
    g=fig.add_gridspec(2,1,height_ratios=[1.45,1],left=.065,right=.99,top=.985,bottom=.055,hspace=.045)
    low=g[1].subgridspec(1,2,width_ratios=[1.0,1.06],wspace=.15)
    return [g[0],low[0],low[1]]


def coverage_by_size3(fig,ss,ctx):
    ax=fig.add_subplot(ss);d=f1.COV;y=np.arange(len(d));off=.13
    for i,(key,col,lab,mk) in enumerate([('systems','share_systems','Systems','o'),('people','share_people','People served','s')]):
        yy=y+(i-.5)*2*off
        ax.hlines(yy,.25,d[col]*100,color=C[key],lw=1.3)
        ax.plot(d[col]*100,yy,ls='',marker=mk,ms=5,color=C[key],mec=INK,mew=.45,label=lab)
    ax.set_yticks(y,d.cls);ax.set_ylim(len(d)-.55,-.5);ax.set_xscale('log');ax.set_xlim(.25,125)
    ax.set_xticks([.3,1,3,10,30,100],['0.3','1','3','10','30','100']);ax.minorticks_off()
    ax.axhline(3.5,color=MUTED,lw=.7,ls='--');ax.legend(loc='upper right',labelspacing=.35,handlelength=1.2)
    style_axis(ax,xlabel='Monitored in UCMR5 (%)',ylabel='People served per system',tick_size=10.5,label_size=11.5)


def efficiency_round3(pop,est,logy=False,ylim=None,yticks=None):
    old=f4.eff_panel(pop,est,logy=logy,ylim=ylim,yticks=yticks)
    def draw(fig,ss,ctx):
        old(fig,ss,ctx);ax=fig.axes[-1]
        # An open, larger diamond leaves the matched Neyman triangle visible.
        for line in ax.lines:
            if line.get_marker() == 'D':
                line.set_markerfacecolor('none');line.set_markersize(6.3)
                line.set_markeredgewidth(.7)
        for axis in fig.axes:
            lg=axis.get_legend()
            if lg:
                for handle,text in zip(lg.legend_handles,lg.get_texts()):
                    if text.get_text() == 'ANCHOR':
                        handle.set_markerfacecolor('none');handle.set_markersize(6.3)
        for t in list(ax.texts):
            if ' · ' in t.get_text():t.remove()
        if est == 'systems':
            ax.text(.03,.97,pop.capitalize(),ha='left',va='top',transform=ax.transAxes,fontsize=10.5)
            if not ctx['standalone']:ax.set_xlabel('')
        if pop == 'medium' or ctx['standalone']:
            ax.set_ylabel('System efficiency' if est == 'systems' else 'Population efficiency')
        ax.xaxis.label.set_fontsize(11.5);ax.yaxis.label.set_fontsize(11.5)
    return draw


def layout4_round3(fig):
    g=fig.add_gridspec(4,1,height_ratios=[.15,.93,.98,1.10],left=.065,right=.99,top=.99,bottom=.045,hspace=.045)
    hs=f4.method_handles()
    for h in hs:
        if h.get_label() == 'ANCHOR':h.set_markerfacecolor('none');h.set_markersize(6.3)
    legend_strip(fig,g[0],hs,4,handlelength=1.55,columnspacing=.75,labelspacing=.25)
    top=g[1].subgridspec(1,2,wspace=.16);mid=g[2].subgridspec(1,2,wspace=.16)
    low=g[3].subgridspec(1,2,width_ratios=[1,1.12],wspace=.14)
    return [top[0],top[1],mid[0],mid[1],low[0],low[1]]


def coverage_round3(fig,ss,ctx):
    f4.panel_coverage(fig,ss,ctx);ax=fig.axes[-1]
    labels={'Population PPS':'PPS','ML-assisted GREG':'GREG','SRS + difference':'SRS + diff.',
            'Neyman allocation':'Neyman','Neyman + difference':'Neyman + diff.'}
    ax.set_yticks(ax.get_yticks(),[labels.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()])


def draft_export():
    """Drafts only: fs._export applies current outer-boundary alignment and QA."""
    fs.MIN_PT=10.5;fs.LETTER_PT=12
    plt.rcParams.update({'font.size':10.5,'axes.labelsize':11.5,'xtick.labelsize':10.5,
      'ytick.labelsize':10.5,'legend.fontsize':10.5,'legend.title_fontsize':10.5,'axes.labelweight':'normal'})
    out=OUTPUT/'内部核查'/'第三轮草稿';out.mkdir(parents=True,exist_ok=True)
    tasks=[('Fig1_overview_round3',(190,145),[coverage_map3,coverage_by_size3,f1.panel_c],layout1_round3),
      ('Fig4_validation_round3',(190,182),[
       efficiency_round3('medium','systems',ylim=(0,2.65),yticks=[0,1,2]),
       efficiency_round3('large','systems',ylim=(0,2.35),yticks=[0,1,2]),
       efficiency_round3('medium','people',ylim=(0,3.15),yticks=[0,1,2,3]),
       efficiency_round3('large','people',logy=True,ylim=(.3,70),yticks=[.3,1,3,10,30]),
       f4.panel_pareto,coverage_round3],layout4_round3)]
    for stem,size,builders,layout in tasks:
        fig=plt.figure(figsize=np.array(size)/25.4)
        for b,ss,key in zip(builders,layout(fig),'abcdef'):draw_module(fig,ss,key,b)
        result=fs._export(fig,out,stem,True)
        print(stem,result['n_unexpected'],len(result['label_overlaps']),flush=True)


if __name__ == '__main__':draft_export()
