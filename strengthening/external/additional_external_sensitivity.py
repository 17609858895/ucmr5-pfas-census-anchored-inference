from strengthening import paths
"""Schedule sensitivity and follow-up-selection diagnostics; frozen external outcomes.

Run after verify_wa_source.py, calendar_sensitivity.py, and
followup_selection_diagnostic.py. Does not alter the primary count-window cohort.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as stm
from sklearn.metrics import roc_auc_score, average_precision_score

P = paths.EXTERNAL_RESULTS
A = P.parent
O = paths.EXTERNAL_RESULTS
ROOT = paths.WORKDIR
B = 2000

def ci(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return np.percentile(x, [2.5, 97.5]) if len(x) else [np.nan, np.nan]

def auc_counts(y, score, counts):
    order = np.argsort(score, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    pos = np.add.reduceat(counts[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(counts[:, order] * (1-y[order]), starts, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (pos*(np.cumsum(neg, axis=1)-.5*neg)).sum(axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))

def join_predictions(s):
    for folder, stem, prefix in [
        ('region_holdout', 'spatial_predictions_frame', 'cold'),
        ('operational_coordinate_diagnostic', 'operational_coordinate_predictions', 'available'),
        ('region_coordinate_diagnostic', 'spatial_predictions_frame', 'cold_diag')]:
        x = pd.read_parquet(paths.SPATIAL_RESULTS/folder/(stem+'.parquet'))
        cols = [c for c in ['attributes','geography','neighbours','coordinates','coordinates_neighbours'] if c in x]
        s = s.merge(x[['PWSID']+cols].rename(columns={c:prefix+'_'+c for c in cols}), on='PWSID', validate='many_to_one')
    return s

frame = pd.read_parquet(paths.DERIVED/'small_frame_anchor.parquet')
spacing = []
for state, locfile in [('MI','MI_strict_locations.csv'),('WA','WA_source_verified_locations.csv')]:
    loc = pd.read_csv(O/locfile)
    loc = loc[loc.span_days.ge(loc.k.map({2:150,4:240}))]
    values = [c for c in ['y_full','y_threshold4','y_R1','y_R2','y_R12','y_quantization_lo','y_quantization_hi','y_PFOS_allmax','y_PFOS_allsum'] if c in loc]
    agg = {c:'max' for c in values}
    agg.update({c:'sum' for c in ['c_full','c_R1','c_R2','c_R12']})
    g = loc.groupby('PWSID', as_index=False).agg(agg).merge(frame,on='PWSID',validate='one_to_one')
    g['external_state'] = state; g['schedule_sensitivity'] = 'first_k_minimum_spacing'
    g.to_csv(O/(state+'_spacing_sensitivity_systems.csv'),index=False)
    spacing.append(g)

calendar = pd.read_csv(O/'calendar_aligned_systems.csv')
calendar['schedule_sensitivity'] = 'calendar_2023_2025'
s = join_predictions(pd.concat([calendar]+spacing,ignore_index=True))
s.to_csv(O/'schedule_sensitivity_predictions.csv',index=False)

old = pd.read_csv(paths.WORKDIR/'results/external_validation_followup.csv')
factor = old[old['item'].eq('national recalibration factor k')].iloc[0]
models = ['f','available_neighbours','cold_neighbours','available_coordinates_neighbours','cold_diag_coordinates_neighbours']
rows = []
for (state, rule), g in s.groupby(['external_state','schedule_sensitivity']):
    g = g.reset_index(drop=True); n = len(g); y = g.y_full.to_numpy(float)
    for kind in ['system','spatial_0.5degree']:
        rng = np.random.default_rng(813+n)
        if kind == 'system':
            counts = rng.multinomial(n, np.full(n,1/n), size=B).astype(float); nb=n
        else:
            block = np.where(g.lat.notna()&g.lon.notna(), np.floor(g.lat/.5).astype('Int64').astype(str)+'|'+np.floor(g.lon/.5).astype('Int64').astype(str), 'missing|'+g.PWSID)
            code = pd.factorize(block)[0]; nb = code.max()+1
            counts = rng.multinomial(nb,np.full(nb,1/nb),size=B)[:,code].astype(float)
        nt = counts.sum(axis=1)
        for model in models:
            f = g[model].to_numpy(float); auc = roc_auc_score(y,f)
            assert abs(auc_counts(y,f,np.ones((1,n)))[0]-auc)<1e-12
            aci = ci(auc_counts(y,f,counts)); oci = ci(counts@y/nt)
            errci = ci(counts@(f-y)/nt); bci = ci(counts@((f-y)**2)/nt)
            rows.append({'state':state,'schedule':rule,'resampling':kind,'model':model,'n':n,'positive':int(y.sum()),'observed':y.mean(),'observed_lo':oci[0],'observed_hi':oci[1],'predicted':f.mean(),'auc':auc,'auc_lo':aci[0],'auc_hi':aci[1],'pr_auc':average_precision_score(y,f),'brier':np.mean((f-y)**2),'brier_lo':bci[0],'brier_hi':bci[1],'prediction_error':np.mean(f-y),'prediction_error_lo':errci[0],'prediction_error_hi':errci[1],'bootstrap_units':nb,'full_cost':g.c_full.sum(),'R1_cost':g.c_R1.sum(),'R12_cost':g.c_R12.sum(),'R1_missed':int((g.y_full-g.y_R1).sum()),'R12_missed':int((g.y_full-g.y_R12).sum()),'R12_early_stopped_fraction':float((g.c_R12<g.c_full).mean()),'R12_cost_saving':float(1-g.c_R12.sum()/g.c_full.sum()),'R12_sensitivity':float(g.y_R12.sum()/g.y_full.sum())})
        f = g.f.to_numpy(float)
        # These are factor-only intervals, not uncertainty for external-cohort truth.
        rows.append({'state':state,'schedule':rule,'resampling':'national_factor_only','model':'f_national_ratio','n':n,'positive':int(y.sum()),'observed':y.mean(),'predicted':np.clip(f*factor.value,0,1).mean(),'predicted_lo':np.clip(f*factor.lo,0,1).mean(),'predicted_hi':np.clip(f*factor.hi,0,1).mean()})
pd.DataFrame(rows).drop_duplicates().to_csv(O/'schedule_sensitivity_metrics.csv',index=False)

# Descriptive eligibility association, with time opportunity restriction and
# adjustment for observed screening date, population, and system source class.
screen = pd.read_csv(O/'initial_screening_systems.csv',parse_dates=['first_date'])
screen = screen[screen.early_index_366day_opportunity.eq(1)]
adjusted=[]
for state,g in screen.groupby('external_state'):
    g=g.copy(); X=pd.DataFrame({'initial_detect4':g.det_first_screen.astype(float),'log10_population':np.log10(g.P.clip(lower=1)),'system_surface':g.sw.astype(float),'index_elapsed_years':(g.first_date-pd.Timestamp('2023-01-01')).dt.days/365.25})
    blocks = np.where(g.lat.notna()&g.lon.notna(),np.floor(g.lat/.5).astype('Int64').astype(str)+'|'+np.floor(g.lon/.5).astype('Int64').astype(str),'missing|'+g.PWSID)
    fit = stm.GLM(g.strict_eligible.astype(float),stm.add_constant(X),family=stm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':blocks})
    b=fit.params['initial_detect4']; se=fit.bse['initial_detect4']
    adjusted.append({'state':state,'n':len(g),'spatial_blocks':len(set(blocks)),'adjusted_OR_detect4':np.exp(b),'lo':np.exp(b-1.959963984540054*se),'hi':np.exp(b+1.959963984540054*se),'adjustment':'log10 population, system source class, initial-date elapsed years; first date allows 366-day follow-up; 0.5-degree spatial-cluster robust CI; descriptive association'})
pd.DataFrame(adjusted).to_csv(O/'followup_eligibility_adjusted_OR.csv',index=False)
(O/'schedule_sensitivity_protocol.json').write_text(json.dumps({'calendar':'Earliest complete calendar year in 2023–2025; first paired sample in each half-year for GW or quarter for surface. No outcome selection. No annual regulatory-compliance claim.','spacing':'Restrict original selected first-k event sets to spans >=150 days GW / >=240 days surface, still <=366. No re-selection of dates.','source_class':'WA official per-source inventory; MI current system source class proxy.','intervals':'2000 PWS or 0.5-degree administrative-centroid block draws; fixed predictions, selected-cohort uncertainty only. Ratio intervals: original national calibration-factor uncertainty only.','selection_model':'Descriptive clustered logistic eligibility association; covariate adjustment is not causal identification.'},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("model=='f' and resampling=='system'")[['state','schedule','n','positive','observed','predicted','auc','auc_lo','auc_hi','R12_missed','R12_cost_saving']].to_string(index=False))
print(pd.DataFrame(adjusted).to_string(index=False))
