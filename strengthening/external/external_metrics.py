from .metric_core import auc_counts, exact_auc_label_bounds
from strengthening import paths
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from scipy.stats import rankdata
P=paths.EXTERNAL_RESULTS;O=paths.EXTERNAL_RESULTS;B=2000
S=pd.read_csv(O/'external_schedule_predictions.csv')
MODELS=['f']+[c for c in S if c.startswith(('cold_','available_'))]
def ci(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a,[2.5,97.5])] if len(a) else [np.nan,np.nan]
def count_draws(g,kind,seed):
    rng=np.random.default_rng(seed);n=len(g)
    if kind=='system':return rng.multinomial(n,np.full(n,1/n),size=B).astype(float),n
    block=np.where(g.lat.notna()&g.lon.notna(),np.floor(g.lat/.5).astype('Int64').astype(str)+'|'+np.floor(g.lon/.5).astype('Int64').astype(str),'missing|'+g.PWSID)
    code=pd.factorize(block)[0];nb=code.max()+1
    return rng.multinomial(nb,np.full(nb,1/nb),size=B)[:,code].astype(float),int(nb)
rows=[];paired=[];bins=[]
for state,g0 in S.groupby('external_state'):
 for subset,g in [('all',g0),('groundwater',g0[g0.sw.eq(0)])]:
  g=g.reset_index(drop=True);y=g.y_full.to_numpy(float);a=g.P.to_numpy(float)
  boot={}
  for kind in ['system','spatial_0.5degree']:
   counts,nb=count_draws(g,kind,317+len(g));nt=counts.sum(axis=1)
   for model in MODELS:
    f=g[model].to_numpy(float);pp=np.clip(f,1e-8,1-1e-8)
    av=auc_counts(y,f,counts);assert abs(auc_counts(y,f,np.ones((1,len(g))))[0]-roc_auc_score(y,f))<1e-12
    brier=counts@((pp-y)**2)/nt;err=counts@(pp-y)/nt;obs=counts@y/nt;pred=counts@pp/nt
    num=counts@(a*y);aden=counts@a;popobs=num/aden;poppred=counts@(a*pp)/aden
    row={'state':state,'subset':subset,'resampling':kind,'bootstrap_draws':B,'blocks_or_units':nb,'model':model,'n':len(g),'positive':int(y.sum()),'observed':y.mean(),'observed_lo':ci(obs)[0],'observed_hi':ci(obs)[1],'predicted':pp.mean(),'predicted_lo':ci(pred)[0],'predicted_hi':ci(pred)[1],'prediction_error':np.mean(pp-y),'prediction_error_lo':ci(err)[0],'prediction_error_hi':ci(err)[1],'auc':roc_auc_score(y,f),'auc_lo':ci(av)[0],'auc_hi':ci(av)[1],'valid_auc_bootstraps':int(np.isfinite(av).sum()),'pr_auc':average_precision_score(y,f),'brier':np.mean((pp-y)**2),'brier_lo':ci(brier)[0],'brier_hi':ci(brier)[1],'log_loss':log_loss(y,pp,labels=[0,1]),'population_observed':np.average(y,weights=a),'population_predicted':np.average(pp,weights=a),'population_observed_lo':ci(popobs)[0],'population_observed_hi':ci(popobs)[1],'population_prediction_error_lo':ci(poppred-popobs)[0],'population_prediction_error_hi':ci(poppred-popobs)[1]}
    rows.append(row);boot[(kind,model)]=(av,brier)
  for kind in ['system','spatial_0.5degree']:
   for warm,cold in [('available_neighbours','cold_neighbours'),('available_geography','cold_geography'),('available_coordinates','cold_diag_coordinates'),('available_coordinates_neighbours','cold_diag_coordinates_neighbours')]:
    aa,bb=boot[(kind,warm)];ac,bc=boot[(kind,cold)]
    paired.append({'state':state,'subset':subset,'resampling':kind,'available_model':warm,'cold_model':cold,'auc_available_minus_cold':roc_auc_score(y,g[warm])-roc_auc_score(y,g[cold]),'auc_difference_lo':ci(aa-ac)[0],'auc_difference_hi':ci(aa-ac)[1],'brier_cold_minus_available':np.mean((g[cold]-y)**2-(g[warm]-y)**2),'brier_difference_lo':ci(bc-bb)[0],'brier_difference_hi':ci(bc-bb)[1]})
  for model in ['f','available_neighbours','cold_neighbours']:
   order=np.argsort(g[model].to_numpy(),kind='stable');binid=np.empty(len(g),int);binid[order]=np.minimum(np.arange(len(g))*4//len(g),3)
   for k in range(4):
    h=g[binid==k];bins.append({'state':state,'subset':subset,'model':model,'predicted_quartile':k+1,'n':len(h),'positives':int(h.y_full.sum()),'predicted_mean':h[model].mean(),'observed_mean':h.y_full.mean()})
pd.DataFrame(rows).to_csv(O/'strict_external_metrics_bootstrap.csv',index=False)
pd.DataFrame(paired).to_csv(O/'paired_information_layer_differences.csv',index=False)
pd.DataFrame(bins).to_csv(O/'external_calibration_quartiles.csv',index=False)

sens=[]
for state,g in S.groupby('external_state'):
 for end in ['y_full','y_threshold4','y_quantization_lo','y_quantization_hi','y_PFOS_allmax','y_PFOS_allsum']:
    if g[end].isna().all():continue
    yy=g[end].to_numpy(int)
    for model in ['f','available_neighbours','cold_neighbours']:
      sens.append({'state':state,'endpoint':end,'model':model,'n':len(g),'positives':int(yy.sum()),'prevalence':yy.mean(),'auc':roc_auc_score(yy,g[model]),'brier':np.mean((g[model]-yy)**2)})
 g=g[g.external_state.eq('WA')]
 if len(g):
  for model in ['f','available_neighbours','cold_neighbours']:
   lo,hi=exact_auc_label_bounds(g[model].to_numpy(),g.y_quantization_lo.to_numpy(int),g.y_quantization_hi.to_numpy(int))
   sens.append({'state':'WA','endpoint':'hypothetical_nearest_integer_rounding_scenario_exact_AUC_range','model':model,'n':len(g),'positives':int(g.y_quantization_lo.sum()),'positive_upper':int(g.y_quantization_hi.sum()),'auc_lower':lo,'auc_upper':hi,'ambiguous_labels':int((g.y_quantization_hi-g.y_quantization_lo).sum())})
pd.DataFrame(sens).to_csv(O/'threshold_duplicate_and_quantization_sensitivity.csv',index=False)
(O/'metric_protocol.json').write_text(json.dumps({'bootstrap_draws':B,'random_seed_rule':'317+cohortn','unit_bootstrap':'resamplePWSIDs','spatial_bootstrap':'resample0.5degreeadministrative-centroidgridblocks;missingsingletons','interval_scope':'Conditionalcohort-compositionuncertainty,frozenpredictions;notnationalorstateprobability-samplingCI','calibration_bins':'fourequal-countpredictedriskbins(nooutcomeused)','quantization_status':'HypotheticalnearestintegerroundingassumptionNOTdocumentedstateprecision:reportedscenariorangesarenotidentifiedtruebounds.'},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("subset=='all' and resampling=='system' and model in ['f','available_neighbours','cold_neighbours']")[['state','model','n','positive','observed','predicted','auc','auc_lo','auc_hi','brier']].to_string(index=False))
