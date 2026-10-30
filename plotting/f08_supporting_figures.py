from f_common import *
R=OUTPUT/'results'; G=pd.read_csv(R/'grid_summary.csv'); SP=pd.read_csv(R/'small_projection.csv');TL=pd.read_csv(R/'two_level_census.csv');SM=pd.read_csv(R/'two_level_small.csv')
def two(fig):
 g=fig.add_gridspec(1,2,left=.06,right=.99,top=.98,bottom=.12,wspace=.20);return [g[0],g[1]]
def axis_with_legend(fig,ss,handles,ncol=2):
 g=ss.subgridspec(2,1,height_ratios=[.3,1],hspace=.05);legend_strip(fig,g[0],handles,ncol,labelspacing=.25,handlelength=1.3);return fig.add_subplot(g[1])
IV=[('wald','Wald',PAL[0]),('logit_t','Logit-t',PAL[1]),('wilson','Wilson',PAL[6]),('kg','Korn–Graubard',PAL[3])]
def interval_panel(width=False):
 def draw(fig,ss,ctx):
  d=G[(G.prior=='geo+nb')&(G.label=='ANCHOR (Poisson + PPI++)')]
  ax=axis_with_legend(fig,ss,[Line2D([],[],color=c,marker='o',ms=4,label=lab) for _,lab,c in IV]);ns=[100,150,300,600,1000]
  for key,lab,c in IV:
   x=d.assign(v=d['wid_'+key]/d.wid_wald if width else d['cov_'+key]).groupby('n').v
   ax.plot(ns,x.mean().reindex(ns),color=c,marker='o',ms=4,mec=INK,mew=.4);ax.fill_between(ns,x.min().reindex(ns),x.max().reindex(ns),color=c,alpha=.15,lw=0)
  ax.axhline(1 if width else .95,color=MUTED,lw=.8,ls='--');ax.set_xscale('log');ax.set_xticks([100,300,1000],['100','300','1,000']);ax.minorticks_off()
  if width:ax.set_ylim(.85,1.18)
  else:ax.set_ylim(.90,.98)
  style_axis(ax,xlabel='Systems sampled',ylabel='Width relative to Wald' if width else '95% interval coverage')
 return draw
DES=[('SRS','hajek','SRS',PAL[6]),('EPA-actual','hajek','Reconstructed cells',PAL[0]),('Neyman-H10','hajek','Neyman',PAL[1]),('Neyman-H10','ppi','Neyman + difference',PAL[7]),('ANCHOR','ppi','ANCHOR',PAL[3]),('dual-rho0.25','ppi','Dual',PAL[4])]
def small_panel(est):
 def draw(fig,ss,ctx):
  d=SP[(SP.n==800)&(SP.estimand==est)];sc=['fitted','auc-match','fitted+sd'];hs=[Line2D([],[],color=c,marker='o',ms=4,label=l) for _,_,l,c in DES]
  ax=axis_with_legend(fig,ss,hs,2)
  for de,ee,l,c in DES:
   g=d[(d.design==de)&(d.estimator==ee)].groupby('scenario').eff_vs_epa;v=g.mean().reindex(sc);sd=g.std().reindex(sc)
   ax.errorbar(range(3),v,yerr=sd,marker='o',ms=4,color=c,ecolor=c,lw=1.2,capsize=2,mec=INK,mew=.4)
  ax.axhline(1,color=MUTED,ls='--',lw=.8);ax.set_xticks(range(3),['Fitted','AUC matched','Extra variation']);ax.set_xlim(-.2,2.2);ax.set_ylim(bottom=0);ax.text(.03,.96,'Systems' if est=='systems' else 'People served',transform=ax.transAxes,va='top')
  style_axis(ax,ylabel='Efficiency vs reconstructed cells')
 return draw
def feature_panel(est):
 def draw(fig,ss,ctx):
  keys=[('Neyman-H5 + PPI++ (hybrid)','Neyman, 5 strata'),('Neyman-H10 + PPI++ (hybrid)','Neyman, 10 strata'),('ANCHOR (Poisson + PPI++)','ANCHOR')]
  ax=axis_with_legend(fig,ss,[Patch(fc=PAL[1],label='Base + geography'),Patch(fc=PAL[3],label='+ Neighbours')]);d=G[(G.n==300)&(G.estimand==est)&(G.population=='medium')]
  for j,(pr,c) in enumerate([('geo',PAL[1]),('geo+nb',PAL[3])]):
   v=[d[(d.prior==pr)&(d.label==key)].rel_eff.iloc[0] for key,l in keys]
   ax.bar(np.arange(3)+(j-.5)*.36,v,width=.34,color=c,ec=INK,lw=.5)
  ax.axhline(1,color=MUTED,ls='--',lw=.8);ax.set_xticks(range(3),[l for _,l in keys],rotation=20,ha='right');ax.set_ylim(0,3);ax.text(.03,.96,'Systems' if est=='systems' else 'People served',transform=ax.transAxes,va='top');style_axis(ax,ylabel='Efficiency vs SRS')
 return draw
def fraction_panel(eff=False):
 def draw(fig,ss,ctx):
  d=TL[(TL.population=='large')&(TL.estimand=='people')];ns=[150,300,600];cc=[PAL[1],PAL[3],PAL[4]]
  ax=axis_with_legend(fig,ss,[Line2D([],[],color=c,marker='o',label=str(n)) for n,c in zip(ns,cc)],3)
  for n,c in zip(ns,cc):
   vals=[];los=[];his=[]
   for q in [.1,.2,.3]:
    x=d[(d.n0==n)&(d.strategy==f'ANCHOR · R1 + two-phase q={q}')].set_index('seed')
    if eff:
     ref=d[(d.n0==n)&(d.strategy=='ANCHOR · full')].set_index('seed').mse;v=ref/x.mse
    else:v=x.cov_kg
    vals.append(v.mean());los.append(v.min());his.append(v.max())
   ax.plot([.1,.2,.3],vals,color=c,marker='o',ms=4,mec=INK,mew=.4);ax.fill_between([.1,.2,.3],los,his,color=c,alpha=.16,lw=0)
  ax.axhline(1 if eff else .95,color=MUTED,ls='--',lw=.8);ax.set_xticks([.1,.2,.3]);ax.set_xlim(.085,.315);style_axis(ax,xlabel='Full-schedule fraction q',ylabel='Efficiency vs ANCHOR, full' if eff else '95% interval coverage')
 return draw
def dual_panel(cost=False):
 def draw(fig,ss,ctx):
  d=SM[SM.strategy=='Dual · R12 + two-phase'];refs=SM[SM.strategy.str.startswith('REFERENCE')].set_index(['scenario','seed','estimand']).mse
  d=d.join(refs.rename('mse_ref'),on=['scenario','seed','estimand']);d['need']=d.cost_mean*d.mse/d.mse_ref
  ax=axis_with_legend(fig,ss,[Patch(fc=C[e],label=l) for e,l in [('systems','Systems'),('people','People served')]])
  for j,e in enumerate(['systems','people']):
   g=d[d.estimand==e].groupby('scenario')['need' if cost else 'cov_kg'];v=g.mean().reindex(['fitted','auc-match']);lo=g.min().reindex(['fitted','auc-match']);hi=g.max().reindex(['fitted','auc-match']);xx=np.arange(2)+(j-.5)*.3
   ax.bar(xx,v,width=.28,color=C[e],ec=INK,lw=.5);ax.errorbar(xx,v,yerr=[v-lo,hi-v],fmt='none',color=INK,lw=.8,capsize=2)
  ax.set_xticks([0,1],['Fitted','AUC matched']);ax.set_xlim(-.5,1.5)
  if cost:ax.set_ylim(0,1550)
  else:ax.set_ylim(.90,.97);ax.axhline(.95,color=MUTED,ls='--',lw=.8)
  style_axis(ax,ylabel='Projected equal-MSE samples' if cost else '95% interval coverage')
 return draw
if __name__=='__main__':
 for name,builders in [('FigS1_interval_sensitivity',[interval_panel(),interval_panel(True)]),('FigS2_small_frame_designs',[small_panel('systems'),small_panel('people')]),('FigS3_feature_allocation',[feature_panel('systems'),feature_panel('people')]),('FigS4_completion_fraction',[fraction_panel(),fraction_panel(True)]),('FigS5_dual_design',[dual_panel(True),dual_panel()])]:
  save_modules(builders,name,name,two,figsize_mm=(180,87),sizes_mm=[(90,85),(90,85)])
