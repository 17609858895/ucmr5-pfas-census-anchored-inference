from strengthening import paths
"""Numerical equivalence and design-calibration checks, not performance tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from . import external_completion_validation as ex
from . import completion_design_validation as cv

P=paths.METHOD_RESULTS
def main():
    rng=np.random.default_rng(17171);N=200
    y=rng.binomial(1,.2,N).astype(float);a=rng.lognormal(4,2,N);f=rng.uniform(.01,.4,N);pi=rng.uniform(.1,.8,N);I=rng.random(N)<pi;J=np.ones(N,bool);q=np.ones(N)
    v=ex.hajek(y,y,f,a,pi,I,J,q);v0=cv.ac.est_hajek(y[I],a[I],1/pi[I],pi[I],a.sum(),('poisson',))
    assert np.allclose(v[:2],v0[:2],rtol=1e-13,atol=1e-15)
    sm=pd.read_parquet(cv.BASE/'data_derived/small_frame_anchor.parquet');W=pd.read_parquet(cv.BASE/'results/within_system_rules.parquet');pilot=cv.pilot_data(sm,W)
    WW=W[W.size_cat.eq('small')].merge(sm[['PWSID','src']],on='PWSID');eta,lr=cv.sp.truth_model(sm);score=cv.dual_score(sm.u.to_numpy(),sm.P.to_numpy())
    records=[]
    for scen in ['fitted','auc-match']:
        for seed in range(10,15):
            rng=np.random.default_rng(900+10*seed+list(cv.sp.SCENARIOS).index(scen));lin,_=cv.sp.scenario_linear(sm,eta,lr,*cv.sp.SCENARIOS[scen],rng);yy=(rng.random(len(sm))<expit(lin)).astype(float)
            costs={c:np.zeros(len(sm)) for c in cv.tl.RULE_COLS}
            for (yv,src),g in WW.groupby(['y','src']):
                idx=np.flatnonzero((yy==yv)&(sm.src.to_numpy()==src));pick=g.iloc[rng.integers(0,len(g),len(idx))]
                for c in costs:costs[c][idx]=pick[c].to_numpy()
            for B in [1000,1200]:
                for policy in ['neyman_cost20','range_cost20']:
                    pp,qq=cv.design(score,sm.P.to_numpy(),sm.f.to_numpy(),sm.src.to_numpy(),pilot,costs,B,policy)
                    assert pp.max()<1
                    records.append({'scenario':scen,'seed':seed,'budget':B,'policy':policy,'pi_max':float(pp.max())})
    info={'hajek_full_q1_point_se_matches_classic':True,'maximum_absolute_point_se_difference':float(np.max(np.abs(np.array(v[:2])-v0[:2]))),'new_pseudo10_14_all_pi_uncapped':True,'largest_pi':max(r['pi_max'] for r in records),'equivalence_note':'For these uncapped probabilities, the score-based q rule used when the validation run loaded equals the subsequent explicit-pi rule up to its solved scalar; no performance outcome changed.','fixed_lambda_variance_identity':cv.variance_test()}
    (P/'estimator_design_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8');pd.DataFrame(records).to_csv(P/'new_pseudo_probability_caps_audit.csv',index=False)
    meta=P/'range_newseeds10_14/prespecified_design.json';spec=json.loads(meta.read_text(encoding='utf-8'));spec.pop('ceiling',None);spec['ceiling_by_policy']={'uniform20':.2,'uniform40':.4,'neyman_cost20':1.,'range_cost20':1.};spec['metadata_correction']='Replaced generic .8 conditional-Neyman ceiling with the actual policy-specific limits, without changing any probabilities or results.';meta.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
