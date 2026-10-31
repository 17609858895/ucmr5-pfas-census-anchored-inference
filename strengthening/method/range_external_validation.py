from strengthening import paths
"""Cost-matched modeled/range completion; all prior external results retained."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import completion_design_validation as cv
from . import external_completion_validation as ex
P=paths.METHOD_RESULTS;D=P/'range_external';D.mkdir(exist_ok=True)
POLICIES=['neyman_cost20','range_cost20'];R=1000
def main():
    cv.POLICIES=POLICIES.copy()
    inp=pd.read_csv(ex.COHORT,dtype={'PWSID':str});cold=pd.read_parquet(ex.COLD)[['PWSID','neighbours']].rename(columns={'neighbours':'cold_f'})
    data=inp.merge(cold,on='PWSID',how='left',validate='one_to_one');assert data.cold_f.notna().all()
    sm=pd.read_parquet(cv.BASE/'data_derived/small_frame_anchor.parquet');W=pd.read_parquet(cv.BASE/'results/within_system_rules.parquet');refcache={}
    out=[];detail=[];start=time.perf_counter()
    protocol={'reference_file':'external_design_by_setting.csv','cohort_sha256':hashlib.sha256(ex.COHORT.read_bytes()).hexdigest(),'R':R,'policies':cv.POLICIES,'qmin':.1,'qmax':1.,'pilot_incremental_completion_cost_fraction':.2,'never_external_missed_label_in_policy':True,'point_known_pi_q_variance_identity':cv.variance_test(),'outcome_inspection_history':'Michigan and prior pseudo0–9 informed mechanism. Prior Washington policies executed but uninspected before this rule; no blind-Washington claim.'}
    (D/'range_external_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    for si,(state,g) in enumerate(data.groupby('external_state',sort=True)):
        P_=g.P.to_numpy(float);y=g.y_full.to_numpy(float);srcs=g.src.to_numpy()
        for rule in ex.RULES:
            costs={'cost_full':g.c_full.to_numpy(float),'cost_R12':g['c_'+rule].to_numpy(float),'yt_R12':g['y_'+rule].to_numpy(float)};pilot=cv.pilot_data(sm,W,rule)
            for regime in ex.REGIMES:
                f=g.cold_f.to_numpy() if regime=='cold_region_sd_proxy' else g.f.to_numpy();u=g.u.to_numpy() if regime=='operational_learned_error' else np.sqrt(np.clip(f,1e-5,1-1e-5)*(1-np.clip(f,1e-5,1-1e-5)))
                setting={'kind':'independent_external_finite_population','population':state,'scenario':regime,'rule':rule,'N':len(g),'positives':int(y.sum()),'missed_full_population':int((y-costs['yt_R12']).sum()),'regime':regime}
                budgets=[v*costs['cost_full'].mean() for v in ex.BUDGET_EQ];oo,dd=cv.run_setting(y,f,u,P_,srcs,costs,pilot,setting,R,budgets,20000+si*100)
                for r in oo:
                    key=(state,r['budget'])
                    if key not in refcache:
                        pi=cv.tl.budget_probs(np.ones(len(g)),costs['cost_full'],r['budget'])
                        refcache[key]=cv.reference(y,f,P_,costs,pi,setting,20000+si*100+int(r['budget']),R,'reference_equal_full')
                    reference=next(v for v in refcache[key] if v['estimand']==r['estimand'])
                    r['mse_ratio_to_reference']=r['mse']/reference['mse'];r['budget_full_equivalents']=r['budget']/costs['cost_full'].mean()
                out+=oo;detail+=dd;pd.DataFrame(out).to_csv(D/'range_external_by_setting.csv',index=False)
                print(state,rule,regime,round(time.perf_counter()-start,1),flush=True)
    pd.DataFrame(detail).to_csv(D/'range_external_draws.csv',index=False)
    (D/'range_external_run.json').write_text(json.dumps({'pass':True,'settings':len(out),'elapsed_seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
