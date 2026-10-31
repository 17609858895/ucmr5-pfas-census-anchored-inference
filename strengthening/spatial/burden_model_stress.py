from strengthening import paths
"""Frozen-prediction burden stress using the unchanged eight-cell estimator."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
P=paths.SPATIAL_RESULTS;R=paths.WORKDIR
sys.path.insert(0,str(paths.ANALYSIS_CODE))
import anchor_core as ac
d=pd.read_parquet(paths.DERIVED/'small_frame_anchor.parquet')
op=pd.read_parquet(P/'operational_coordinate_diagnostic/operational_coordinate_predictions.parquet')
cold=pd.read_parquet(P/'region_holdout/spatial_predictions_with_diagnostic.parquet')
models=['attributes','geography','neighbours','coordinates','coordinates_neighbours']
base=d[['PWSID','y_mcl','cell','P','f']].copy().rename(columns={'f':'frozen_production'})
base=base.merge(op[['PWSID']+models].rename(columns={m:'available_'+m for m in models}),on='PWSID',how='left',validate='one_to_one')
base=base.merge(cold[['PWSID']+models].rename(columns={m:'cold_'+m for m in models}),on='PWSID',how='left',validate='one_to_one')
assert len(base)==56591 and base.iloc[:,4:].notna().all().all()
cells=base.cell.astype('category').cat.codes.to_numpy();observed=base.y_mcl.notna().to_numpy()
Nh=np.bincount(cells).astype(float);nh=np.bincount(cells[observed]);pi=(nh/Nh)[cells[observed]];w=1/pi
design=('strat',cells[observed],Nh,nh);y=base.loc[observed,'y_mcl'].to_numpy(float)
rows=[]
for estimand in ['systems','people']:
 a=np.ones(len(base)) if estimand=='systems' else base.P.to_numpy(float);A=a.sum();aa=a[observed]
 for model in ['frozen_production']+['available_'+m for m in models]+['cold_'+m for m in models]:
  f=base[model].to_numpy(float);ll=float(ac._lam(y,f[observed],aa,w))
  theta,se,df=ac.est_ppi(y,f[observed],aa,w,pi,A,np.sum(a*f),design)
  lo,hi=ac.intervals_vec([theta],[se],[df],[len(y)])['kg'];ne=ac._neff(theta,se,len(y));kgne=max(ne*(t.ppf(.975,len(y)-1)/t.ppf(.975,df))**2,1)
  rows.append(dict(estimand=estimand,model=model,N=len(base),n=len(y),prediction_mean=np.sum(a*f)/A,theta=theta,lambda_value=ll,se=se,df=df,kg_lo=lo[0],kg_hi=hi[0],variance_effective_n=ne,kg_effective_n=kgne,kish_weight_effective_n=w.sum()**2/np.sum(w*w)))
 theta,se,df=ac.est_hajek(y,aa,w,pi,A,design);lo,hi=ac.intervals_vec([theta],[se],[df],[len(y)])['kg']
 rows.append(dict(estimand=estimand,model='direct_poststratified',N=len(base),n=len(y),prediction_mean=np.nan,theta=theta,lambda_value=np.nan,se=se,df=df,kg_lo=lo[0],kg_hi=hi[0],variance_effective_n=ac._neff(theta,se,len(y)),kish_weight_effective_n=w.sum()**2/np.sum(w*w)))
out=pd.DataFrame(rows)
for target,expected in [('systems',.07858819255204613),('people',.06721613038072392)]:
 actual=out[(out.estimand==target)&(out.model=='frozen_production')].theta.iloc[0];assert abs(actual-expected)<1e-14,(target,actual,expected)
out.to_csv(P/'burden_model_stress.csv',index=False)
point=out[out.model!='direct_poststratified'].groupby('estimand').agg(prediction_min=('prediction_mean','min'),prediction_max=('prediction_mean','max'),corrected_min=('theta','min'),corrected_max=('theta','max'),lambda_min=('lambda_value','min'),lambda_max=('lambda_value','max'))
(P/'burden_model_stress_validation.json').write_text(json.dumps({'frozen_reproduced_exactly':True,'frozen_system_prevalence':.07858819255204613,'frozen_population_prevalence':.06721613038072392,'working_cells':8,'n':970,'N':56591,'bootstrap_performed':False,'interval':'Sameworkingstratifiedvariance/degrees/KornGraubardhelper as frozenoriginal, predictionsfixed. Unknownhistoricalselectionnotincluded.','models':11,'targets':2,'source_helper_sha256':hashlib.sha256(Path(ac.__file__).read_bytes()).hexdigest(),'ranges':point.to_dict(orient='index')},indent=2),encoding='utf-8')
print(out[['estimand','model','prediction_mean','theta','lambda_value','kg_lo','kg_hi']].to_string(index=False))
print(point)
