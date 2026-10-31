from strengthening import paths
from pathlib import Path
import numpy as np,pandas as pd
from .summarize_spatial import metrics
P=paths.SPATIAL_RESULTS
a=pd.read_parquet(P/'region_buffer50_reference/spatial_predictions_frame.parquet');b=pd.read_parquet(P/'region_buffer50/spatial_predictions_frame.parquet')
a=a[(a.size_cat=='small')&a.y_mcl.notna()&a.lat.notna()&a.lon.notna()].copy();b=b.set_index('PWSID').loc[a.PWSID]
y=a.y_mcl.to_numpy().astype(int);g=a.block.to_numpy().astype(int)-1;w=np.ones(len(a));rng=np.random.default_rng(20261031)
C=np.vstack([np.ones(10),rng.multinomial(10,np.full(10,.1),2000)])
rows=[];vals={}
for regime,p in [('unbuffered_geo_complete_source',a.neighbours.to_numpy()),('buffer50',b.neighbours.to_numpy())]:
 point,v=metrics(y,p,w,g,C);rows.append(dict(regime=regime,**point));vals[regime]=v
pd.DataFrame(rows).to_csv(P/'buffer_matched_metrics.csv',index=False)
delta=[]
for metric in ['auc','within_region_auc','brier','calibration_bias']:
 v=vals['buffer50'][metric]-vals['unbuffered_geo_complete_source'][metric]
 delta.append(dict(metric=metric,comparison='buffer50_minus_matched_unbuffered',difference=v[0],lo=np.nanpercentile(v[1:],2.5),hi=np.nanpercentile(v[1:],97.5)))
pd.DataFrame(delta).to_csv(P/'buffer_paired_contrasts.csv',index=False)
print(pd.DataFrame(rows)[['regime','n','positives','auc','auc_lo','auc_hi','within_region_auc','brier']].to_string(index=False))
print(pd.DataFrame(delta).to_string(index=False))
