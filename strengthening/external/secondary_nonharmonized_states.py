from strengthening import paths
"""Endpoint-mismatched MN/NH ranking sensitivity with all UCMR5 IDs excluded.

These aggregate/source-water endpoints cannot validate the retained-mean
PFOA/PFOS target or state population prevalence. No outcome-based fitting.
"""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from . import build_external_cohorts as c
P=paths.EXTERNAL_RESULTS;O=paths.EXTERNAL_RESULTS;A=P.parent
EXT=c.EXT

mn=pd.read_excel(EXT/'minnesota_mdh_20261025/mn_cws_pfas_results.xlsx',sheet_name='Table',dtype=str)
mn=mn[~mn.Notes.fillna('').eq('Purchased Water')].copy();mn['PWSID']='MN'+mn.pwsid.str.zfill(7)
for a in ['PFOA_Min','PFOA_Max','PFOS_Min','PFOS_Max']:mn[a]=pd.to_numeric(mn[a].replace('ND','0'),errors='coerce')
assert not mn[['PFOA_Min','PFOA_Max','PFOS_Min','PFOS_Max']].isna().any().any()
mn['y_max4']=mn[['PFOA_Max','PFOS_Max']].max(axis=1).ge(4).astype(int)
mn['y_min4']=mn[['PFOA_Min','PFOS_Min']].max(axis=1).ge(4).astype(int)
assert not mn.PWSID.duplicated().any()
mn=mn[['PWSID','y_max4','y_min4']];mn['external_state']='MN'
nh=pd.read_csv(EXT/'newhampshire_nhdes_20261025/nh_pws_pfas_sources.csv',dtype=str)
assert set(nh.fedr_mcl.unique())<=set(['Y','N']) and set(nh.e_mcl.unique())<=set(['Y','N'])
nh['PWSID']='NH'+nh.pwsid.str.split('-').str[0].str.zfill(7)
nh=nh.groupby('PWSID',as_index=False).agg(y_federal_flag=('fedr_mcl',lambda x:int(x.eq('Y').any())),y_state_flag=('e_mcl',lambda x:int(x.eq('Y').any())))
nh['external_state']='NH'
S=pd.concat([mn,nh],ignore_index=True).merge(c.sm,on='PWSID',validate='one_to_one')
S=S[~S.PWSID.isin(c.excluded)].copy()
for folder,stem,prefix in [('region_holdout','spatial_predictions_frame','cold'),('operational_coordinate_diagnostic','operational_coordinate_predictions','available')]:
 x=pd.read_parquet(paths.SPATIAL_RESULTS/folder/(stem+'.parquet'))[['PWSID','neighbours']].rename(columns={'neighbours':prefix+'_neighbours'})
 S=S.merge(x,on='PWSID',validate='one_to_one')
S.to_csv(O/'secondary_nonharmonized_independent_systems.csv',index=False)
rows=[]
for state,g in S.groupby('external_state'):
 endpoints=['y_max4','y_min4'] if state=='MN' else ['y_federal_flag','y_state_flag']
 for endpoint in endpoints:
  y=g[endpoint].to_numpy(int);rng=np.random.default_rng(1011+len(g));n=len(g)
  draws=rng.integers(0,n,(2000,n));block=np.where(g.lat.notna()&g.lon.notna(),np.floor(g.lat/.5).astype('Int64').astype(str)+'|'+np.floor(g.lon/.5).astype('Int64').astype(str),'missing|'+g.PWSID)
  codes=pd.factorize(block)[0];nb=codes.max()+1;bc=rng.multinomial(nb,np.full(nb,1/nb),size=2000)[:,codes]
  for model in ['f','available_neighbours','cold_neighbours']:
   f=g[model].to_numpy(float)
   unit=[];sp=[]
   for idx,w in zip(draws,bc):
    if np.unique(y[idx]).size==2:unit.append(roc_auc_score(y[idx],f[idx]))
    if len(np.unique(y[w>0]))==2:sp.append(roc_auc_score(y,f,sample_weight=w))
   for kind,vals in [('system',unit),('spatial_0.5degree',sp)]:
    lo,hi=np.percentile(vals,[2.5,97.5])
    rows.append({'state':state,'endpoint':endpoint,'scope':'secondary endpoint-mismatched ranking only; not retained means, calibrated probabilities or representative prevalence','n':n,'positives':int(y.sum()),'observed_cohort_fraction':y.mean(),'model':model,'auc':roc_auc_score(y,f),'auc_lo':lo,'auc_hi':hi,'resampling':kind,'pr_auc':average_precision_score(y,f)})
pd.DataFrame(rows).to_csv(O/'secondary_nonharmonized_state_metrics.csv',index=False)
(O/'secondary_state_protocol.json').write_text(json.dumps({'UCMR5_exclusion':'All 10,313 unique PWSIDs with any record excluded. Same fully independent cohort for counts and AUC.','MN':'Static CWS per-system minima/maxima, purchased-water rows excluded, ND=0. Cannot reconstruct retained location means or sample timing. Threshold 4 ng/L sensitivity only.','NH':'Source-water federal/state exceedance flags; no analyte-specific concentration or finished-water retained mean. Ranking against broad PFAS flags only.','interpretation':'Secondary sensitivity, not evidence of endpoint-identical external validation or population burden. Never pool these outcomes with MI/WA count-matched means.','bootstrap':'2000 PWS or 0.5-degree administrative-centroid spatial-block draws; frozen scores; cohort-conditional intervals.'},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("model=='f' and resampling=='spatial_0.5degree'").to_string(index=False))
