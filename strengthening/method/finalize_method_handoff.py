from strengthening import paths
"""Freeze and audit all prespecified results; no numerical policy selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from . import completion_design_validation as cv

P=paths.METHOD_RESULTS
INTERVALS=cv.ac.INTERVALS
LABELS={'reference_equal_full':'SRS full HT','srs_full_hajek':'SRS full Hajek','srs_full_difference':'SRS full difference','dual_full':'Dual full difference','dual_full_hajek':'Dual full Hajek','dual_two_phase_hajek':'Dual two-phase Hajek','uniform20':'Dual + uniform 20%','uniform40':'Dual + uniform 40%','system20':'Dual + system Neyman 20%','dual20':'Dual + dual Neyman 20%','uncorrected':'Dual stopped without correction','pps_full_hajek':'PPS full Hajek','pps_two_phase_hajek':'PPS two-phase Hajek','system_anchor_two_phase':'System ANCHOR + uniform 20%','neyman5_full_hajek':'Five-stratum full Hajek','neyman5_full_difference':'Five-stratum full difference','neyman5_full_difference_fixed1':'Five-stratum full difference, lambda=1','neyman_cost20':'Dual + cost Neyman 20%','range_cost20':'Dual + range balance 20%'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_complete(base,expected):
    files=[base/'baseline/external_design_by_setting.csv',base/'range/range_external_by_setting.csv',base/'classical/external_classical_by_setting.csv']
    frames=[]
    for f in files:
        d=pd.read_csv(f);d['source_result_file']=str(f.relative_to(P));frames.append(d)
    d=pd.concat(frames,ignore_index=True);assert len(d)==expected,(len(d),expected)
    # Correct labels only: the existing fixed1 implementation always used lambda=1.
    d.loc[d.policy.str.endswith('fixed1'),'coefficient']='one'
    # The classical helper did not retain fitted lambdas; do not report dummy 0.
    d.loc[d.policy.eq('neyman5_full_difference'),'lambda_mean']=np.nan
    d.loc[d.coefficient.eq('ratio'),'lambda_mean']=np.nan
    for name in INTERVALS:
        col='coverage_'+name
        if col in d:d[col+'_mc_se']=np.sqrt(d[col]*(1-d[col])/d.draws)
    keys=['population','rule','regime','budget_full_equivalents','estimand']
    for policy,suffix in [('srs_full_hajek','SRS_Hajek'),('neyman5_full_difference','Neyman_full')]:
        ref=d[d.policy.eq(policy)][keys+['mse','mse_mc_se']].rename(columns={'mse':'ref_mse_'+suffix,'mse_mc_se':'ref_mse_mc_se_'+suffix})
        assert not ref.duplicated(keys).any()
        d=d.merge(ref,on=keys,validate='many_to_one')
        d['mse_ratio_to_'+suffix]=d.mse/d['ref_mse_'+suffix]
    d['policy_label']=d.policy.map(LABELS).fillna(d.policy)
    d['shared_sample_for_two_estimands']=~d.policy.str.startswith('neyman5_')
    d['allocation_scope']=np.where(d.shared_sample_for_two_estimands,'one joint sample, both targets','target-specific full-schedule Neyman reference')
    d.loc[d.policy.eq('uniform20')&d.coefficient.eq('one'),'policy_label']='Dual + uniform 20%, lambda=1'
    assert not d.duplicated(keys+['policy','coefficient']).any()
    assert np.allclose(d.expected_cost,d.budget,rtol=1e-8)
    return d,files
def neyman_audit(path):
    plans=json.loads(path.read_text(encoding='utf-8'))
    for v in plans:
        assert min(v['lower_nh'])>=2 and min(v['upper_nh'])>=2
        assert 0<=v['probability_upper_plan']<=1
        assert np.isclose(v['expected_cost_exact'],v['budget'])
        assert np.isclose((1-v['probability_upper_plan'])*v['lower_expected_cost']+v['probability_upper_plan']*v['upper_expected_cost'],v['budget'])
    return {'allocation_plans':len(plans),'all_minimum_nh_at_least_two':True,'all_expected_costs_matched':True,'largest_absolute_budget_error':max(abs(v['expected_cost_exact']-v['budget']) for v in plans)}
def main():
    # Standardized tree preserves the separately recorded run outputs.
    base=P/'external_v2_bundle';base.mkdir(exist_ok=True)
    mapping={'baseline':P/'external_v2','range':P/'range_external_v2','classical':P/'external_classical_v2'}
    for k,src in mapping.items():
        dst=base/k;dst.mkdir(exist_ok=True)
        name={'baseline':'external_design_by_setting.csv','range':'range_external_by_setting.csv','classical':'external_classical_by_setting.csv'}[k]
        (dst/name).write_bytes((src/name).read_bytes())
    for f in [P/'external_v2/external_design_run.json',P/'range_external_v2/range_external_run.json',P/'external_classical_v2/run.json']:
        assert json.loads(f.read_text(encoding='utf-8'))['pass']
    d,files=load_complete(base,960);d['cohort_version']='count_window_source_verified_v2';d.to_csv(P/'external_design_complete_v2.csv',index=False)
    x=d[d.rule.eq('R12')&d.regime.eq('operational_learned_error')&d.budget_full_equivalents.eq(300)].copy();assert len(x)==80;x.to_csv(P/'external_primary_R12_B300.csv',index=False)
    chosen=['srs_full_hajek','neyman5_full_difference','pps_full_hajek','system_anchor_two_phase','uniform20','uniform40','neyman_cost20','range_cost20']
    figs=x[x.policy.isin(chosen)&~(x.policy.eq('uniform20')&x.coefficient.eq('one'))].copy();assert len(figs)==32
    figs.to_csv(P/'external_Fig7_candidates.csv',index=False)
    cal,calfiles=load_complete(P/'calendar_sensitivity',80);cal['cohort_version']='calendar_aligned_sensitivity';cal.to_csv(P/'external_calendar_R12_B100.csv',index=False)
    cohort=pd.read_csv(P/'external_cohort_v2_frozen.csv',dtype={'PWSID':str});cold=pd.read_parquet(paths.SPATIAL_RESULTS/'region_holdout/spatial_predictions_frame.parquet')[['PWSID','neighbours']];cohort=cohort.merge(cold,on='PWSID',validate='one_to_one');assert len(cohort)==2138
    points=[]
    for st,g in cohort.groupby('external_state'):
        for reg in ['operational_learned_error','cold_region_sd_proxy']:
            f=g.f.to_numpy() if reg=='operational_learned_error' else g.neighbours.to_numpy()
            for target,a in [('systems',np.ones(len(g))),('people',g.P.to_numpy())]:
                truth=np.average(g.y_full,weights=a);m=np.average(f,weights=a)
                points.append({'population':st,'regime':reg,'estimand':target,'N':len(g),'positives':g.y_full.sum(),'truth':truth,'model_only':m,'model_only_error':m-truth,'model_only_squared_error':(m-truth)**2,'R1_no_completion':np.average(g.y_R1,weights=a),'R12_no_completion':np.average(g.y_R12,weights=a),'full_cost_sum':g.c_full.sum(),'R12_cost_sum':g.c_R12.sum(),'R12_missed_systems':(g.y_full-g.y_R12).sum()})
    pd.DataFrame(points).to_csv(P/'external_frozen_prediction_diagnostics.csv',index=False)
    new=pd.read_csv(P/'range_newseeds10_14/completion_small_by_seed.csv');new.groupby(['scenario','budget','estimand','policy','coefficient'],dropna=False)[['mse','mse_ratio_to_reference','coverage_kg','width_kg','variance_ratio','expected_full_fraction','zero_missed_full_fraction']].mean().reset_index().to_csv(P/'fresh_pseudo10_14_summary.csv',index=False)
    audit={'primary':neyman_audit(P/'external_classical_v2/neyman_cost_matched_plans.json'),'calendar':neyman_audit(P/'calendar_sensitivity/classical/neyman_cost_matched_plans.json')};(P/'neyman_cost_plan_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    frozen_source_hashes=json.loads((P/'prespecified_design.json').read_text(encoding='utf-8'))['source_sha256']
    assert frozen_source_hashes==cv.HASHES
    for f,oldhash in frozen_source_hashes.items():assert sha(next(p for p in cv.FILES if p.name==f))==oldhash
    results=[P/'external_design_complete_v2.csv',P/'external_primary_R12_B300.csv',P/'external_Fig7_candidates.csv',P/'external_calendar_R12_B100.csv',P/'external_frozen_prediction_diagnostics.csv',P/'fresh_pseudo10_14_summary.csv']
    manifest={'frozen_analysis_complete':True,'new_candidates_after_range':False,'draws_per_setting':1000,'primary_rows':len(d),'calendar_rows':len(cal),'primary_cohort_sha256':sha(P/'external_cohort_v2_frozen.csv'),'calendar_cohort_sha256':sha(P/'calendar_sensitivity/calendar_cohort_frozen.csv'),'primary_counts':{'MI':[1283,25],'WA':[855,84]},'original_source_hashes_unchanged':cv.HASHES,'result_sha256':{r.name:sha(r) for r in results},'code_sha256':{r.name:sha(r) for r in Path(__file__).resolve().parent.glob('*.py')},'estimator_audit':json.loads((P/'estimator_design_audit.json').read_text()),'neyman_plan_audit':audit,'metadata_normalization':['fixed1 coefficient is one; numerical implementation already used lambda=1','fitted lambda was not retained for classical Neyman helper and is left missing, not reported as zero'],'all_negative_and_positive_policies_retained':True}
    (P/'analysis_manifest_final.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['population','estimand','policy','coefficient','bias','bias_mc_se','mse_ratio_to_SRS_Hajek','mse_ratio_to_Neyman_full','coverage_kg','coverage_kg_mc_se','width_kg','variance_ratio','zero_missed_full_fraction']
    (P/'external_primary_results_table.md').write_text(x[cols].round(6).to_markdown(index=False),encoding='utf-8')
    print(json.dumps({'primary':len(d),'primary_display':len(x),'figure_candidates':len(figs),'calendar':len(cal),'files':[r.name for r in results]},indent=2))
if __name__=='__main__':main()
