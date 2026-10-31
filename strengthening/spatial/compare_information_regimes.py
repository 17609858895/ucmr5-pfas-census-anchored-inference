from strengthening import paths
from pathlib import Path
import json
import numpy as np,pandas as pd
from .summarize_spatial import metrics
from . import spatial_transfer as st
P=paths.SPATIAL_RESULTS;R=paths.WORKDIR
frame=st.setup(pd.read_parquet(paths.DERIVED/'frame_systems.parquet'))
source=frame[frame.size_cat.isin(['medium','large'])&frame.y_mcl.notna()]
small=frame[frame.size_cat=='small'].copy();small['cell']=small.sw.astype(str)+'|'+pd.cut(small['pop'],[-1,100,500,1000,3300],labels=['a','b','c','d']).astype(str)
small['working_weight']=small.cell.map(small.groupby('cell').size()/small[small.y_mcl.notna()].groupby('cell').size())
op=pd.read_parquet(P/'operational_coordinate_diagnostic/operational_coordinate_predictions.parquet')
cold=pd.read_parquet(P/'region_holdout/spatial_predictions_with_diagnostic.parquet')
frozen=pd.read_parquet(paths.DERIVED/'small_frame_anchor.parquet')[['PWSID','f']]
d=small[small.y_mcl.notna()][['PWSID','block','y_mcl','pop','working_weight']].merge(op[['PWSID']+[c for c in op if c in st.SETS or c.startswith('coordinates')]],on='PWSID',validate='one_to_one')
d=d.merge(frozen,on='PWSID',validate='one_to_one').merge(cold[['PWSID','attributes','geography','neighbours','coordinates','coordinates_neighbours']].rename(columns={m:'cold_'+m for m in ['attributes','geography','neighbours','coordinates','coordinates_neighbours']}),on='PWSID',validate='one_to_one')
assert len(d)==970
y=d.y_mcl.to_numpy().astype(int);g=d.block.to_numpy().astype(int)-1
rng=np.random.default_rng(20261031);C=np.vstack([np.ones(10),rng.multinomial(10,np.full(10,.1),2000)])
models=['f','attributes','geography','neighbours','coordinates','coordinates_neighbours']
rows=[];paired=[];saved={}
for weighting,w in {'systems':np.ones(len(d)),'people':d['pop'].clip(lower=25).to_numpy(),'working_poststrat_systems':d.working_weight.to_numpy(),'working_poststrat_people':d.working_weight.to_numpy()*d['pop'].clip(lower=25).to_numpy()}.items():
 for model in models:
  for regime in ['available','cold'] if model!='f' else ['frozen_original_available']:
   col=('cold_'+model if regime=='cold' else model);p=d[col].to_numpy();point,values=metrics(y,p,w,g,C)
   rows.append(dict(regime=regime,weighting=weighting,model=model,**point));saved[(weighting,regime,model)]=values
 for model in models[1:]:
  for metric in ['auc','within_region_auc','brier','calibration_bias']:
   diff=saved[(weighting,'available',model)][metric]-saved[(weighting,'cold',model)][metric]
   paired.append(dict(weighting=weighting,model=model,contrast='available_minus_cold',metric=metric,difference=diff[0],lo=np.nanpercentile(diff[1:],2.5),hi=np.nanpercentile(diff[1:],97.5)))
 for regime in ['available','cold']:
  for a,b in [('neighbours','geography'),('coordinates_neighbours','coordinates')]:
   for metric in ['auc','brier']:
    diff=saved[(weighting,regime,a)][metric]-saved[(weighting,regime,b)][metric]
    paired.append(dict(weighting=weighting,model=a+'_minus_'+b,contrast=regime,metric=metric,difference=diff[0],lo=np.nanpercentile(diff[1:],2.5),hi=np.nanpercentile(diff[1:],97.5)))
pd.DataFrame(rows).to_csv(P/'information_regime_metrics.csv',index=False)
pd.DataFrame(paired).to_csv(P/'information_regime_paired_contrasts.csv',index=False)
d.to_parquet(P/'small_information_regime_predictions.parquet',index=False)
# Mechanistic availability diagnostics use the same administrative ZIP centroids.
lab=small[small.y_mcl.notna()].copy();a=st.nb_features(lab,source)
b=pd.DataFrame(np.nan,index=lab.index,columns=st.NB)
for region,t in lab.groupby('block'):b.loc[t.index]=st.nb_features(t,source[source.block!=region])
support=lab[['PWSID','block','y_mcl']].copy();support['available_n50']=a.nb_n50;support['cold_n50']=b.nb_n50
support.to_csv(P/'small_neighbour_availability.csv',index=False)
support_summary=[]
for region,t in [('all',support),*[(str(r),g) for r,g in support.groupby('block')]]:
 support_summary.append(dict(region=region,n=len(t),available_fraction=t.available_n50.fillna(0).gt(0).mean(),cold_fraction=t.cold_n50.fillna(0).gt(0).mean(),missing_coordinate_n=int(t.available_n50.isna().sum())))
pd.DataFrame(support_summary).to_csv(P/'neighbour_availability_summary.csv',index=False)
print(pd.DataFrame(rows).query("weighting=='systems'")[['regime','model','auc','auc_lo','auc_hi','within_region_auc','brier','mean_prediction']].to_string(index=False))
print(support_summary[0])

# Aggregate ROC paths support source-only public figure reproduction.
from sklearn.metrics import roc_curve
roc_rows=[]
for regime in ['available','cold']:
 for model in ['attributes','geography','neighbours']:
  key=model if regime=='available' else 'cold_'+model
  x,y,_=roc_curve(d.y_mcl,d[key])
  roc_rows.extend({'regime':regime,'model':model,'fpr':a,'tpr':b} for a,b in zip(x,y))
pd.DataFrame(roc_rows).to_csv(P/'plot_roc_curves.csv',index=False)
