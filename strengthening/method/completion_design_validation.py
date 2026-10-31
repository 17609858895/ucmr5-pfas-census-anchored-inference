from .estimator_core import dual_score, estimate, variance_test
from strengthening import paths
"""Prespecified completion policies; new pseudo-populations and measured populations.

The conditional-Neyman policy is an adaptation of established two-phase
allocation, not a claim of a new general optimum. All policy choices and
floors are fixed before generation of new validation pseudo-populations.
"""
from pathlib import Path
import sys,json,hashlib,pickle,time,itertools,argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import skew
from threadpoolctl import threadpool_limits
OUT=paths.METHOD_RESULTS;ROOT=paths.WORKDIR
BASE=paths.WORKDIR
sys.path.insert(0,str(paths.ANALYSIS_CODE))
import s20_small_projection as sp
import s24_two_level as tl
import anchor_core as ac
sys.stdout.reconfigure(encoding='utf-8')
R=1000;BUDGETS=[1000,1200];SEEDS=list(range(5,10))
POLICIES=['uniform20','uniform40','system20','dual20']
FLOOR=.1;CEILING=.8
FILES=[BASE/'data_derived/small_frame_anchor.parquet',BASE/'results/within_system_rules.parquet',paths.ANALYSIS_CODE/'s24_two_level.py',paths.ANALYSIS_CODE/'anchor_core.py']
HASHES={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}


def capped_cost_design(score,P,h,dc,costs,B,policy):
    """Solve scale of capped pi and the exact conditional pi-dependent q rule.

    This is calibration within a fixed first-stage ordering, not a claim that
    first- and second-stage probabilities jointly minimize a convex objective.
    """
    g=np.maximum(1,P/P.mean()) if policy=='range_cost20' else np.sqrt(.5*(1+(P/P.mean())**2)*h/dc)
    def given_scale(s):
        pi=np.minimum(1,s*score)
        shape=g/np.maximum(pi,1e-15);shape/=np.median(shape)
        lo,hi=0.,1.
        def value(k):
            q=np.clip(k*shape,FLOOR,1.)
            return np.sum(pi*q*dc)/np.sum(pi*dc),q
        while value(hi)[0]<.2:hi*=2
        for _ in range(44):
            mid=(lo+hi)/2
            if value(mid)[0]<.2:lo=mid
            else:hi=mid
        frac,q=value((lo+hi)/2)
        total=np.sum(pi*(q*costs['cost_full']+(1-q)*costs['cost_R12']))
        return total,pi,q,frac
    lo=0.;hi=B/max(np.sum(score*(.2*costs['cost_full']+.8*costs['cost_R12'])),1e-12)
    for _ in range(35):
        if given_scale(hi)[0]>=B:break
        hi*=2
    else:raise ValueError('Budget infeasible with the fixed pilot completion-cost fraction')
    for _ in range(58):
        mid=(lo+hi)/2
        if given_scale(mid)[0]<B:lo=mid
        else:hi=mid
    total,pi,q,frac=given_scale((lo+hi)/2)
    assert np.isclose(total,B,rtol=1e-9) and np.isclose(frac,.2,atol=1e-9)
    # Uncapped q units have equal range score, or the Neyman analogue.
    mask=(q>FLOOR+1e-8)&(q<1-1e-8)
    if mask.any():assert np.ptp((g/(pi*q))[mask])<1e-7*max(1,np.mean((g/(pi*q))[mask]))
    return pi,q

def design(score,P,f,src,pilot,costs,B,policy):
    if policy.startswith('uniform'):
        q=np.full(len(P),.2 if policy=='uniform20' else .4)
        pi=tl.budget_probs(score,q*costs['cost_full']+(1-q)*costs['cost_R12'],B)
    else:
        h=np.clip(f,1e-5,1-1e-5)*np.array([pilot[s]['miss_fraction_smoothed'] for s in src])
        dc=np.array([pilot[s]['incremental_cost_mean'] for s in src]).clip(.05)
        target=(np.ones(len(P)) if policy=='system20' else .5*(1+(P/P.mean())**2))
        cost_constraint=policy.endswith('_cost20')
        ceiling=1. if cost_constraint else CEILING
        if cost_constraint:
            dc=costs['cost_full']*np.array([pilot[s]['incremental_cost_mean']/pilot[s]['full_cost_mean'] for s in src])
            dc=dc.clip(.05)
        shape=(np.maximum(1,P/P.mean()) if policy=='range_cost20' else np.sqrt(target*h/dc))/np.maximum(score,1e-8)
        shape/=np.median(shape)
        def at(k):
            qq=np.clip(k*shape,FLOOR,ceiling)
            pp=tl.budget_probs(score,qq*costs['cost_full']+(1-qq)*costs['cost_R12'],B)
            criterion=(pp*qq*dc).sum()/(pp*dc).sum() if cost_constraint else (pp*qq).sum()/pp.sum()
            return criterion,pp,qq
        lo,hi=0.,1.
        while at(hi)[0]<.2:hi*=2
        for _ in range(45):
            mid=(lo+hi)/2
            if at(mid)[0]<.2:lo=mid
            else:hi=mid
        _,pi,q=at((lo+hi)/2)
        if cost_constraint and pi.max()>1-1e-12:
            pi,q=capped_cost_design(score,P,h,dc,costs,B,policy)
    assert np.isclose(np.sum(pi*(q*costs['cost_full']+(1-q)*costs['cost_R12'])),B,rtol=1e-9)
    return pi,q



def pilot_data(sm,W,rule='R12'):
    lab=W[W.size_cat.eq('small')].merge(sm[['PWSID','src']],on='PWSID')
    out={}
    for src,g in lab.groupby('src'):
        n=int(g.y.sum());miss=int(((g.y-g['yt_'+rule])>0).sum())
        out[src]={'n':len(g),'positives':n,'missed_exceedances':miss,'miss_fraction_smoothed':(miss+.5)/(n+1),'incremental_cost_mean':float((g.cost_full-g['cost_'+rule]).mean()),'full_cost_mean':float(g.cost_full.mean())}
    return out

def summarize(rows,truth):
    x=np.array(rows);err=x[:,0]-truth;intervals=ac.intervals_vec(x[:,0],x[:,1],x[:,2],x[:,3])
    o={'truth':truth,'bias':float(err.mean()),'bias_mc_se':float(err.std(ddof=1)/np.sqrt(len(err))),'mse':float(np.mean(err**2)),'mse_mc_se':float(np.std(err**2,ddof=1)/np.sqrt(len(err))),'empirical_sd':float(err.std(ddof=1)),'mean_variance':float(np.mean(x[:,1]**2)),'mean_se':float(x[:,1].mean()),'variance_ratio':float(np.mean(x[:,1]**2)/np.var(err,ddof=1)),'error_skewness':float(skew(err)),'mean_n':float(x[:,3].mean()),'lambda_mean':float(x[:,4].mean()),'mean_full_n':float(x[:,5].mean()),'mean_missed_full_n':float(x[:,6].mean()),'zero_missed_full_fraction':float(np.mean(x[:,6]==0)),'mean_phase2_variance_fraction':float(np.mean(x[:,9])/np.mean(x[:,1]**2)),'nonnegative_var_ratio':float(x[:,10].mean()/np.var(err,ddof=1))}
    for name,(lo,hi) in intervals.items():
        o['coverage_'+name]=float(np.mean((lo<=truth)&(truth<=hi)));o['width_'+name]=float(np.mean(hi-lo))
    for group,mask in [('zero_correction',x[:,6]==0),('positive_correction',x[:,6]>0)]:
        lo,hi=intervals['kg'];o['coverage_'+group]=float(np.mean((lo[mask]<=truth)&(truth<=hi[mask]))) if mask.any() else np.nan
    return o

def run_setting(y,f,u,P,src,costs,pilot,setting,Rdraws,Bs,seed):
    score=dual_score(u,P);arrs={'systems':np.ones(len(y)),'people':P};out=[];details=[]
    specs=[(p,'tuned') for p in POLICIES]+([('uniform20','one')] if 'uniform20' in POLICIES else [])
    for B in Bs:
        cfg={p:design(score,P,f,src,pilot,costs,B,p) for p in POLICIES}
        rows={(p,m,t):[] for p,m in specs for t in arrs};costrows={p:[] for p in POLICIES}
        rng=np.random.default_rng(830000+seed*100+int(B))
        for r in range(Rdraws):
            U1=rng.random(len(y));U2=rng.random(len(y))
            for policy in POLICIES:
                pi,q=cfg[policy];I=U1<pi;J=U2<q
                c=float(np.where(J[I],costs['cost_full'][I],costs['cost_R12'][I]).sum());costrows[policy].append(c)
                for mode in (['tuned','one'] if policy=='uniform20' else ['tuned']):
                    for target,a in arrs.items():rows[policy,mode,target].append(estimate(y,costs['yt_R12'],f,a,pi,I,J,q,mode))
        for (policy,mode,target),rr in rows.items():
            a=arrs[target];pi,q=cfg[policy];truth=float(np.sum(a*y)/a.sum());s=summarize(rr,truth)
            out.append({**setting,'seed':setting.get('generation_seed_index',setting.get('model_seed',seed)),'draw_seed_index':seed,'budget':B,'policy':policy,'coefficient':mode,'estimand':target,'draws':Rdraws,'expected_full_fraction':float(np.sum(pi*q)/pi.sum()),'q_min':float(q.min()),'q_max':float(q.max()),'expected_cost':float(np.sum(pi*(q*costs['cost_full']+(1-q)*costs['cost_R12']))),'mean_cost':float(np.mean(costrows[policy])),**s})
            x=np.array(rr);lo,hi=ac.intervals_vec(x[:,0],x[:,1],x[:,2],x[:,3])['kg']
            details.extend({**setting,'seed':setting.get('generation_seed_index',setting.get('model_seed',seed)),'draw_seed_index':seed,'budget':B,'policy':policy,'coefficient':mode,'estimand':target,'draw':i,'truth':truth,'estimate':v[0],'se':v[1],'lo':lo[i],'hi':hi[i],'lambda':v[4],'n':v[3],'full_n':v[5],'missed_full_n':v[6],'correction':v[7],'phase2_variance':v[9],'cost':costrows[policy][i]} for i,v in enumerate(x))
    return out,details

def reference(y,f,P,costs,pi,setting,seed,Rdraws,name):
    akeys={'systems':np.ones(len(y)),'people':P};rows={t:[] for t in akeys};cc=[]
    rng=np.random.default_rng(950000+seed);q=np.ones(len(y));J=np.ones(len(y),bool)
    for _ in range(Rdraws):
        I=rng.random(len(y))<pi;cc.append(float(costs['cost_full'][I].sum()))
        for t,a in akeys.items():rows[t].append(estimate(y,y,f,a,pi,I,J,q,'zero'))
    out=[]
    for t,rr in rows.items():
        a=akeys[t];truth=float(np.sum(a*y)/a.sum())
        out.append({**setting,'seed':setting.get('generation_seed_index',setting.get('model_seed',seed)),'draw_seed_index':seed,'budget':float(np.sum(pi*costs['cost_full'])),'policy':name,'coefficient':'zero','estimand':t,'draws':Rdraws,'expected_full_fraction':1.,'q_min':1.,'q_max':1.,'expected_cost':float(np.sum(pi*costs['cost_full'])),'mean_cost':float(np.mean(cc)),**summarize(rr,truth)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['small','measured','test'],default='small');ap.add_argument('--draws',type=int,default=R);args=ap.parse_args()
    test=variance_test();(OUT/'variable_q_variance_identity.json').write_text(json.dumps(test,indent=2),encoding='utf-8')
    if args.mode=='test':print(test);return
    sm=pd.read_parquet(BASE/'data_derived/small_frame_anchor.parquet');W=pd.read_parquet(BASE/'results/within_system_rules.parquet');pilot=pilot_data(sm,W)
    spec={'draws':args.draws,'validation_pseudo_seeds':SEEDS,'policies':POLICIES,'coefficient_ablation':['tuned','one'],'budgets':BUDGETS,'floor':FLOOR,'ceiling_by_policy':{p:(1. if p.endswith('_cost20') else CEILING if not p.startswith('uniform') else .2 if p=='uniform20' else .4) for p in POLICIES},'all_policies_prespecified':True,'historical_0_to_4_consumed_not_holdout':True,'pilot':pilot,'source_sha256':HASHES,'variance_identity':test,'nearest_method_status':'Adaptation of established conditional Neyman/cost-weighted phase-two allocation; no new general optimum claim.'}
    (OUT/'prespecified_design.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.perf_counter();allout=[];detail=[]
    if args.mode=='small':
        eta,lr=sp.truth_model(sm);WW=W[W.size_cat.eq('small')].merge(sm[['PWSID','src']],on='PWSID')
        for scen in ['fitted','auc-match']:
            for seed in SEEDS:
                rng=np.random.default_rng(900+10*seed+list(sp.SCENARIOS).index(scen));lin,_=sp.scenario_linear(sm,eta,lr,*sp.SCENARIOS[scen],rng);y=(rng.random(len(sm))<expit(lin)).astype(float)
                costs={c:np.zeros(len(sm)) for c in tl.RULE_COLS}
                for (yy,src),g in WW.groupby(['y','src']):
                    idx=np.flatnonzero((y==yy)&(sm.src.to_numpy()==src));pick=g.iloc[rng.integers(0,len(g),len(idx))]
                    for c in tl.RULE_COLS:costs[c][idx]=pick[c].to_numpy()
                setting={'kind':'new_pseudo_population','scenario':scen,'population':'small','generation_seed_index':seed}
                o,d=run_setting(y,sm.f.to_numpy(),sm.u.to_numpy(),sm.P.to_numpy(float),sm.src.to_numpy(),costs,pilot,setting,args.draws,BUDGETS,seed+1000*list(sp.SCENARIOS).index(scen))
                cells=sm.cell.astype('category').cat.codes.to_numpy();obs=np.bincount(cells[sm.y_mcl.notna().to_numpy()],minlength=cells.max()+1);Nh=np.bincount(cells,minlength=len(obs));pi=ac.cap_probs((obs/Nh)[cells],800)
                ref=reference(y,sm.f.to_numpy(),sm.P.to_numpy(float),costs,pi,setting,seed+1000*list(sp.SCENARIOS).index(scen),args.draws,'reference_cell_full')
                for r in o:r['mse_ratio_to_reference']=r['mse']/next(v['mse'] for v in ref if v['estimand']==r['estimand'])
                o+=ref
                allout+=o;detail+=d;pd.DataFrame(allout).to_csv(OUT/'completion_small_by_seed.csv',index=False);print(scen,seed,round(time.perf_counter()-start,1),flush=True)
    else:
        F=pd.read_parquet(BASE/'data_derived/frame_features.parquet')
        for pop in ['medium','large']:
            frame=F[F.size_cat.eq(pop)&F.y_mcl.notna()&F.lat.notna()&F['pop'].notna()].reset_index(drop=True)
            ww=W.set_index('PWSID').reindex(frame.PWSID);costs={c:ww[c].to_numpy(float) for c in tl.RULE_COLS}
            src=np.where(frame.sw.to_numpy()==1,'SW','GW')
            for seed in range(5):
                model=pickle.loads((BASE/'data_derived/sim_pops'/f'{pop}_geo+nb_{seed}.pkl').read_bytes());assert np.array_equal(ww.y.to_numpy(float),model['y'])
                setting={'kind':'measured_finite_population','scenario':'observed','population':pop,'model_seed':seed}
                budgets=[int(round(300*costs['cost_full'].mean()))]
                o,d=run_setting(model['y'],model['f'],model['u'],model['P'],src,costs,pilot,setting,args.draws,budgets,10000+100*int(pop=='large')+seed)
                pi=tl.budget_probs(np.ones(len(model['y'])),costs['cost_full'],budgets[0]);ref=reference(model['y'],model['f'],model['P'],costs,pi,setting,10000+100*int(pop=='large')+seed,args.draws,'reference_equal_full')
                for r in o:r['mse_ratio_to_reference']=r['mse']/next(v['mse'] for v in ref if v['estimand']==r['estimand'])
                o+=ref
                allout+=o;detail+=d;pd.DataFrame(allout).to_csv(OUT/'completion_measured_by_seed.csv',index=False);print(pop,seed,round(time.perf_counter()-start,1),flush=True)
    prefix='completion_small' if args.mode=='small' else 'completion_measured'
    pd.DataFrame(detail).to_csv(OUT/(prefix+'_draws.csv'),index=False)
    for p in FILES:assert HASHES[p.name]==hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/(prefix+'_run.json')).write_text(json.dumps({'pass':True,'elapsed_seconds':time.perf_counter()-start,'settings':len(allout),'source_files_unchanged':True},indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
