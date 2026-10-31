from .estimator_core import hajek
from strengthening import paths
"""Prospectively fixed sampling policies evaluated on independent state outcomes."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import completion_design_validation as cv

OUT=paths.METHOD_RESULTS;A2=OUT.parent
COHORT=paths.EXTERNAL_RESULTS/'external_schedule_systems.csv'
COLD=paths.SPATIAL_RESULTS/'region_holdout/spatial_predictions_frame.parquet'
R=1000;BUDGET_EQ=[100,300]
REGIMES=['operational_learned_error','operational_sd_proxy','cold_region_sd_proxy']
RULES=['R1','R12']


def direct_comparators(y,yt,f,u,P,costs,B,setting,seed):
    out=[];N=len(y);score=cv.dual_score(u,P)
    # Strong standard comparators; none uses external outcomes to plan selection.
    for label,scores,q0,full in [('srs_full_hajek',np.ones(N),1.,True),('srs_full_difference',np.ones(N),1.,True),('dual_full_hajek',score,1.,True),('dual_two_phase_hajek',score,.2,False)]:
        q=np.full(N,q0);c=costs['cost_full'] if full else q*costs['cost_full']+(1-q)*costs['cost_R12'];pi=cv.tl.budget_probs(scores,c,B)
        rows={t:[] for t in ['systems','people']};cc=[];rng=np.random.default_rng(830000+seed*100+int(B))
        for _ in range(R):
            I=rng.random(N)<pi;J=rng.random(N)<q;cc.append(float(np.where(J[I],costs['cost_full'][I],costs['cost_R12'][I]).sum()))
            for t,a in [('systems',np.ones(N)),('people',P)]:
                rr=(cv.estimate(y,y,f,a,pi,I,J,q,'tuned') if label=='srs_full_difference' else hajek(y,y if full else yt,f,a,pi,I,J,q))
                rows[t].append(rr)
        for t,rr in rows.items():
            a=np.ones(N) if t=='systems' else P;truth=float(np.sum(a*y)/a.sum())
            out.append({**setting,'seed':seed,'budget':B,'policy':label,'coefficient':'tuned' if label.endswith('difference') else 'ratio','estimand':t,'draws':R,'expected_full_fraction':q0,'q_min':q0,'q_max':q0,'expected_cost':float(np.sum(pi*c)),'mean_cost':float(np.mean(cc)),**cv.summarize(rr,truth)})
    return out

def extra(y,yt,f,u,P,costs,B,setting,seed,uncorrected=False):
    score=cv.dual_score(u,P);cost=costs['cost_R12'] if uncorrected else costs['cost_full']
    pi=cv.tl.budget_probs(score,cost,B);q=np.zeros(len(y)) if uncorrected else np.ones(len(y));rows={t:[] for t in ['systems','people']};cs=[]
    rng=np.random.default_rng(830000+seed*100+int(B))
    for _ in range(R):
        I=rng.random(len(y))<pi;U2=rng.random(len(y));J=U2<q
        cs.append(float(cost[I].sum()))
        for t,a in [('systems',np.ones(len(y))),('people',P)]:rows[t].append(cv.estimate(y,yt if uncorrected else y,f,a,pi,I,J,q,'tuned'))
    out=[]
    for t,rr in rows.items():
        a=np.ones(len(y)) if t=='systems' else P;truth=float(np.sum(a*y)/a.sum())
        out.append({**setting,'seed':seed,'budget':B,'policy':'uncorrected' if uncorrected else 'dual_full','coefficient':'tuned','estimand':t,'draws':R,'expected_full_fraction':0. if uncorrected else 1.,'q_min':float(q.min()),'q_max':float(q.max()),'expected_cost':float(np.sum(pi*cost)),'mean_cost':float(np.mean(cs)),**cv.summarize(rr,truth)})
    return out

def main():
    global OUT
    legacy='--legacy-initial' in sys.argv
    if legacy:OUT=OUT/'first_external_run';OUT.mkdir(exist_ok=True)
    src=pd.read_csv(COHORT,dtype={'PWSID':str});cold=pd.read_parquet(COLD)[['PWSID','neighbours']].rename(columns={'neighbours':'cold_f'})
    data=src.merge(cold,on='PWSID',how='left',validate='one_to_one');assert data.cold_f.notna().all()
    W=pd.read_parquet(cv.BASE/'results/within_system_rules.parquet');sm=pd.read_parquet(cv.BASE/'data_derived/small_frame_anchor.parquet')
    protocol={'R':R,'budget_full_equivalents':BUDGET_EQ,'regimes':REGIMES,'rules':RULES,'paired_rules_and_standard_comparators':not legacy,'no_external_training':True,'no_policy_selected_on_external_outcomes':True,'pilot_rule_calibration_uses_only_historical970':True,'posterior_floor_and_ceiling':[cv.FLOOR,cv.CEILING],'cohort_sha256':hashlib.sha256(COHORT.read_bytes()).hexdigest(),'cold_prediction_sha256':hashlib.sha256(COLD.read_bytes()).hexdigest(),'primary_endpoint':'externalreported-value retained-location mean PFOA/PFOS threshold4.05 after4ng censoring;WAquantization sensitivity separate','cost_assumption':'Schedules and their replay costs treated as fixed in design simulation; global expected-cost calibration uses these fixed costs equally across policies. Unit-specific completion scores do not use target outcomes or target missed-event flags.'}
    (OUT/'external_design_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    out=[];details=[];start=time.perf_counter()
    for si,(state,g) in enumerate(data.groupby('external_state',sort=True)):
        P=g.P.to_numpy(float);y=g.y_full.to_numpy(float);srcs=g.src.to_numpy();N=len(g)
        for ri,rule in enumerate(RULES):
            costs={'cost_full':g.c_full.to_numpy(float),'cost_R12':g['c_'+rule].to_numpy(float),'yt_R12':g['y_'+rule].to_numpy(float)}
            pilot=cv.pilot_data(sm,W,rule)
            for regime in REGIMES:
                f=g.cold_f.to_numpy() if regime=='cold_region_sd_proxy' else g.f.to_numpy()
                u=g.u.to_numpy() if regime=='operational_learned_error' else np.sqrt(np.clip(f,1e-5,1-1e-5)*(1-np.clip(f,1e-5,1-1e-5)))
                # Paired uniform draws across both rules and all information regimes.
                seed=20000+si*100+(ri*10 if legacy else 0)
                B=[b*costs['cost_full'].mean() for b in BUDGET_EQ]
                setting={'kind':'independent_external_finite_population','population':state,'scenario':regime,'rule':rule,'N':N,'positives':int(y.sum()),'missed_full_population':int((y-costs['yt_R12']).sum()),'regime':regime}
                oo,dd=cv.run_setting(y,f,u,P,srcs,costs,pilot,setting,R,B,seed)
                for b in B:
                    pi=cv.tl.budget_probs(np.ones(N),costs['cost_full'],b)
                    ref=cv.reference(y,f,P,costs,pi,setting,seed+int(b),R,'reference_equal_full')
                    for r in ref:r['budget']=b
                    oo+=ref+extra(y,costs['yt_R12'],f,u,P,costs,b,setting,seed,True)+extra(y,y,f,u,P,costs,b,setting,seed,False)
                    if not legacy:oo+=direct_comparators(y,costs['yt_R12'],f,u,P,costs,b,setting,seed)
                for r in oo:
                    ref=next(v for v in oo if v['policy']=='reference_equal_full' and v['estimand']==r['estimand'] and np.isclose(v['budget'],r['budget']))
                    r['mse_ratio_to_reference']=r['mse']/ref['mse'];r['budget_full_equivalents']=r['budget']/costs['cost_full'].mean()
                out+=oo;details+=dd;pd.DataFrame(out).to_csv(OUT/'external_design_by_setting.csv',index=False)
                print(state,rule,regime,'complete',round(time.perf_counter()-start,1),flush=True)
    pd.DataFrame(details).to_csv(OUT/'external_design_draws.csv',index=False)
    assert protocol['cohort_sha256']==hashlib.sha256(COHORT.read_bytes()).hexdigest()
    assert protocol['cold_prediction_sha256']==hashlib.sha256(COLD.read_bytes()).hexdigest()
    (OUT/'external_design_run.json').write_text(json.dumps({'pass':True,'settings':len(out),'draws':R,'elapsed_seconds':time.perf_counter()-start,'all_source_files_unchanged':True},indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
