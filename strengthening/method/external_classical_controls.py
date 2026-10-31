from strengthening import paths
"""Strong simple/stratified controls; no validation-outcome optimization."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import completion_design_validation as cv
from . import external_completion_validation as ex
P=paths.METHOD_RESULTS;D=P/'external_classical';D.mkdir(exist_ok=True)
R=1000

def poisson_control(y,yt,f,u,P_,costs,B,setting,seed,label):
    N=len(y);full=label=='pps_full_hajek';q0=1. if full else .2
    q=np.full(N,q0)
    score=P_ if label.startswith('pps') else cv.ac.mix_scores(u,np.ones(N),.5)
    c=costs['cost_full'] if full else q*costs['cost_full']+(1-q)*costs['cost_R12'];pi=cv.tl.budget_probs(score,c,B)
    rows={t:[] for t in ['systems','people']};cc=[];rng=np.random.default_rng(830000+seed*100+int(B))
    for _ in range(R):
        I=rng.random(N)<pi;J=rng.random(N)<q;cc.append(float(np.where(J[I],costs['cost_full'][I],costs['cost_R12'][I]).sum()))
        for t,a in [('systems',np.ones(N)),('people',P_)]:
            rr=ex.hajek(y,y if full else yt,f,a,pi,I,J,q) if label.startswith('pps') else cv.estimate(y,yt,f,a,pi,I,J,q,'tuned')
            rows[t].append(rr)
    out=[]
    for t,rr in rows.items():
        a=np.ones(N) if t=='systems' else P_;truth=float(np.sum(a*y)/a.sum())
        out.append({**setting,'seed':seed,'budget':B,'policy':label,'coefficient':'ratio' if label.startswith('pps') else 'tuned','estimand':t,'draws':R,'expected_full_fraction':q0,'q_min':q0,'q_max':q0,'expected_cost':float(np.sum(pi*c)),'mean_cost':float(np.mean(cc)),**cv.summarize(rr,truth)})
    return out

def neyman_specs(f,a,cost,B,H=5):
    p=np.clip(f,.001,.999);st=cv.ac.rank_strata(a*f,a*np.sqrt(p*(1-p)),H=H)
    Nh=np.bincount(st,minlength=H).astype(float)
    Sh=np.array([np.sqrt(max(np.mean(a[st==h]**2*p[st==h])-np.mean(a[st==h]*p[st==h])**2,1e-12)) for h in range(H)])
    ch=np.array([cost[st==h].mean() for h in range(H)])
    plans=[]
    for n in range(2*H,len(f)+1):
        nh=cv.ac.neyman_alloc(Nh,Sh,n);c=float(np.sum(nh*ch));plans.append((n,c,nh))
    below=[x for x in plans if x[1]<=B];above=[x for x in plans if x[1]>=B]
    assert below and above,'Budget outside feasible classical-stratum allocations'
    low=max(below,key=lambda x:x[1]);high=min(above,key=lambda x:x[1])
    prob=0. if np.isclose(high[1],low[1]) else (B-low[1])/(high[1]-low[1])
    assert np.isclose((1-prob)*low[1]+prob*high[1],B)
    return st,Nh,low,high,prob

def neyman_control(y,f,P_,costs,B,setting,seed):
    out=[];details=[];N=len(y)
    for ti,(target,a) in enumerate([('systems',np.ones(N)),('people',P_)]):
        st,Nh,low,high,prob=neyman_specs(f,a,costs['cost_full'],B)
        rng=np.random.default_rng(970000+seed*10+ti+int(B));rows={v:[] for v in ['neyman5_full_hajek','neyman5_full_difference','neyman5_full_difference_fixed1']};cc=[];ns=[]
        A=a.sum();truth=float(np.sum(a*y)/A);sumAF=np.sum(a*f)
        for _ in range(R):
            nh=high[2] if rng.random()<prob else low[2];I=cv.ac.draw_stratified(st,nh,rng);pi=(nh/Nh)[st];w=1/pi[I];des=('strat',st[I],Nh,nh)
            cc.append(float(costs['cost_full'][I].sum()));ns.append(int(I.sum()))
            for name,func in [('neyman5_full_hajek',cv.ac.est_hajek),('neyman5_full_difference',cv.ac.est_ppi)]:
                rr=func(y[I],a[I],w,pi[I],A,des) if name.endswith('hajek') else func(y[I],f[I],a[I],w,pi[I],A,sumAF,des)
                rows[name].append((rr[0],rr[1],rr[2],I.sum(),0.,I.sum(),0.,0.,rr[1]**2,0.,rr[1]**2))
            rr=cv.ac.est_ppi(y[I],f[I],a[I],w,pi[I],A,sumAF,des,lam=1.)
            rows['neyman5_full_difference_fixed1'].append((rr[0],rr[1],rr[2],I.sum(),1.,I.sum(),0.,0.,rr[1]**2,0.,rr[1]**2))
        for name,rr in rows.items():
            out.append({**setting,'seed':seed,'budget':B,'policy':name,'coefficient':'ratio' if name.endswith('hajek') else 'one' if name.endswith('fixed1') else 'tuned','estimand':target,'draws':R,'expected_full_fraction':1.,'q_min':1.,'q_max':1.,'expected_cost':B,'mean_cost':float(np.mean(cc)),**cv.summarize(rr,truth)})
        details.append({**setting,'target':target,'budget':B,'strata':5,'Nh':Nh.tolist(),'lower_n':low[0],'lower_expected_cost':low[1],'lower_nh':low[2].tolist(),'upper_n':high[0],'upper_expected_cost':high[1],'upper_nh':high[2].tolist(),'probability_upper_plan':prob,'expected_cost_exact':float((1-prob)*low[1]+prob*high[1]),'selection_of_budget_plan_independent_of_outcomes':True})
    return out,details

def main():
    src=pd.read_csv(ex.COHORT,dtype={'PWSID':str});cold=pd.read_parquet(ex.COLD)[['PWSID','neighbours']].rename(columns={'neighbours':'cold_f'});data=src.merge(cold,on='PWSID',how='left',validate='one_to_one');assert data.cold_f.notna().all()
    refcache={};out=[];plans=[];start=time.perf_counter()
    for si,(state,g) in enumerate(data.groupby('external_state',sort=True)):
        y=g.y_full.to_numpy(float);P_=g.P.to_numpy(float)
        for rule in ex.RULES:
            costs={'cost_full':g.c_full.to_numpy(float),'cost_R12':g['c_'+rule].to_numpy(float),'yt_R12':g['y_'+rule].to_numpy(float)}
            for regime in ex.REGIMES:
                f=g.cold_f.to_numpy() if regime=='cold_region_sd_proxy' else g.f.to_numpy();u=g.u.to_numpy() if regime=='operational_learned_error' else np.sqrt(np.clip(f,1e-5,1-1e-5)*(1-np.clip(f,1e-5,1-1e-5)))
                setting={'kind':'independent_external_finite_population','population':state,'scenario':regime,'rule':rule,'N':len(g),'positives':int(y.sum()),'missed_full_population':int((y-costs['yt_R12']).sum()),'regime':regime}
                for b0 in ex.BUDGET_EQ:
                    B=b0*costs['cost_full'].mean();oo=[]
                    for label in ['pps_full_hajek','pps_two_phase_hajek','system_anchor_two_phase']:
                        oo+=poisson_control(y,costs['yt_R12'],f,u,P_,costs,B,setting,20000+si*100,label)
                    o,pp=neyman_control(y,f,P_,costs,B,setting,20000+si*100);oo+=o;plans+=pp
                    for r in oo:
                        key=(state,r['budget'])
                        if key not in refcache:
                            pi=cv.tl.budget_probs(np.ones(len(g)),costs['cost_full'],r['budget'])
                            refcache[key]=cv.reference(y,f,P_,costs,pi,setting,20000+si*100+int(r['budget']),R,'reference_equal_full')
                        ref=next(v for v in refcache[key] if v['estimand']==r['estimand'])
                        r['mse_ratio_to_reference']=r['mse']/ref['mse'];r['budget_full_equivalents']=b0
                    out+=oo
                pd.DataFrame(out).to_csv(D/'external_classical_by_setting.csv',index=False)
                print(state,rule,regime,round(time.perf_counter()-start,1),flush=True)
    (D/'neyman_cost_matched_plans.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
    (D/'run.json').write_text(json.dumps({'pass':True,'rows':len(out),'elapsed_seconds':time.perf_counter()-start,'cohort_sha256':hashlib.sha256(ex.COHORT.read_bytes()).hexdigest(),'classical_allocations_never_use_external_y':True,'neyman_budget_expected_cost_matched_by_independent_randomization':True},indent=2),encoding='utf-8')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
