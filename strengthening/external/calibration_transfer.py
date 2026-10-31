from strengthening import paths
from pathlib import Path
import json,numpy as np,pandas as pd
P=paths.EXTERNAL_RESULTS;A=P.parent;ROOT=paths.WORKDIR;O=paths.EXTERNAL_RESULTS
sm=pd.read_parquet(paths.DERIVED/'small_frame_anchor.parquet');lab=sm[sm.y_mcl.notna()].copy();S=pd.read_csv(O/'external_schedule_predictions.csv')
Nh=sm.groupby('cell').size();nh=lab.groupby('cell').size();w=lab.cell.map(Nh/nh).to_numpy(float)
k=np.average(lab.y_mcl,weights=w)/np.average(lab.f,weights=w)
old=pd.read_csv(paths.WORKDIR/'results/external_validation_followup.csv');rr=old[old['item'].eq('national recalibration factor k')].iloc[0]
assert np.isclose(k,rr.value,atol=1e-12)
rows=[];factor=[]
for state,g in S.groupby('external_state'):
    f=g.f.to_numpy(float)
    for method,val,lo,hi in [('Uncorrected',f.mean(),np.nan,np.nan),('National ratio',np.clip(f*k,0,1).mean(),np.clip(f*rr.lo,0,1).mean(),np.clip(f*rr.hi,0,1).mean())]:
        rows.append({'state':state,'cohort':'strict independent repeated-sample','n':len(g),'positive':int(g.y_full.sum()),'observed':g.y_full.mean(),'prediction_method':method,'predicted':val,'lo':lo,'hi':hi,'interval_scope':'national calibration-factor uncertainty only; fixed target cohort','observed_population':np.average(g.y_full,weights=g.P),'predicted_population':np.average(np.clip(f*(k if method=='National ratio' else 1),0,1),weights=g.P)})
cal=pd.read_parquet(paths.SPATIAL_RESULTS/'calibration/calibration_by_held_region.parquet')
for (region,regime),c in cal.groupby(['held_region','neighbour_regime']):
    frame=sm[pd.to_numeric(sm.EPA_REGION)!=region]
    c=c.merge(lab[['PWSID','cell']],on='PWSID',how='left',validate='one_to_one')
    assert not c.block.eq(region).any()
    N=frame.groupby('cell').size();n=c.groupby('cell').size();ww=c.cell.map(N/n).to_numpy(float)
    yy=c.y_mcl.to_numpy(float);rng=np.random.default_rng(467+region);idxs={h:np.flatnonzero(c.cell.to_numpy()==h) for h in c.cell.unique()}
    draw=np.concatenate([rng.choice(v,(4000,len(v)),replace=True) for v in idxs.values()],axis=1)
    Y=(ww[draw]*yy[draw]).sum(axis=1)
    for model in ['attributes','geography','neighbours']:
        ff=c[model].to_numpy(float);kk=np.sum(ww*yy)/np.sum(ww*ff);K=Y/((ww[draw]*ff[draw]).sum(axis=1));lo,hi=np.percentile(K,[2.5,97.5])
        factor.append({'held_region':region,'neighbour_regime':regime,'model':model,'calibration_n':len(c),'ratio':kk,'lo':lo,'hi':hi})
        for state,g in S[pd.to_numeric(S.EPA_REGION).eq(region)].groupby('external_state'):
            f=g['cold_'+model].to_numpy(float)
            rows.append({'state':state,'cohort':'strict independent repeated-sample','n':len(g),'positive':int(g.y_full.sum()),'observed':g.y_full.mean(),'prediction_method':'Cold '+model+'; ratio '+regime,'predicted':np.clip(f*kk,0,1).mean(),'lo':np.clip(f*lo,0,1).mean(),'hi':np.clip(f*hi,0,1).mean(),'interval_scope':'outside-held-region calibration-factor uncertainty only; fixed target cohort','observed_population':np.average(g.y_full,weights=g.P),'predicted_population':np.average(np.clip(f*kk,0,1),weights=g.P)})
pd.DataFrame(rows).to_csv(O/'independent_cohort_calibration_transfer.csv',index=False)
pd.DataFrame(factor).to_csv(O/'outside_region_calibration_factors.csv',index=False)
print('National factor',k);print(pd.DataFrame(rows).query("prediction_method in ['Uncorrected','National ratio']")[['state','n','positive','observed','prediction_method','predicted','lo','hi']].to_string(index=False))
