"""Conditional direct-budget re-evaluation of frozen small-frame scenarios.

No LightGBM fitting and no changes to frozen outputs. The original pseudo-truth
and donor assignment sequence is reproduced and checked against frozen truths.
Reference draws extend the original RNG sequence from 600 to 1000 replicates;
the first 600 must reproduce the frozen reference MSE/coverage/mean cost.
"""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.special import expit
import argparse
from project_paths import D, R as RESULTS
import s20_small_projection as sp
import s24_two_level as tl
import anchor_core as ac

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def clone_rng(rng):
    new=np.random.default_rng();new.bit_generator.state=copy.deepcopy(rng.bit_generator.state);return new

def summarize(rows,truth):
    x=np.asarray(rows,float);err=x[:,0]-truth
    lo,hi=ac.intervals_vec(x[:,0],x[:,1],x[:,2],x[:,3])['kg']
    coverage=(lo<=truth)&(truth<=hi);squared=err**2
    return {'truth':float(truth),'bias':float(np.mean(err)),'mse':float(np.mean(squared)),'mse_mc_se':float(np.std(squared,ddof=1)/np.sqrt(len(x))),'cov_kg':float(coverage.mean()),'coverage_mc_se':float(np.sqrt(coverage.mean()*(1-coverage.mean())/len(x))),'mean_interval_width':float(np.mean(hi-lo)),'mean_n':float(x[:,3].mean()),'mean_cost':float(x[:,4].mean()),'cost_sd':float(x[:,4].std(ddof=1)),'draws':len(x)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frame',type=Path,default=D/'small_frame_anchor.parquet')
    ap.add_argument('--rules',type=Path,default=RESULTS/'within_system_rules.parquet')
    ap.add_argument('--frozen-reference',type=Path,default=RESULTS/'two_level_small.csv')
    ap.add_argument('--output',type=Path,default=RESULTS/'review_sensitivity')
    ap.add_argument('--draws',type=int,default=1000)
    ap.add_argument('--original-draws',type=int,default=600)
    ap.add_argument('--budgets',type=int,nargs='+',default=[1000,1200])
    ap.add_argument('--seeds',type=int,default=5)
    args=ap.parse_args()
    if args.draws < args.original_draws or not 1 <= args.seeds <= 5:
        ap.error('--draws must cover --original-draws; --seeds must be 1..5')
    OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
    R=args.draws;OLD_R=args.original_draws;BUDGETS=args.budgets
    sm=pd.read_parquet(args.frame);frozen=pd.read_csv(args.frozen_reference)
    W=pd.read_parquet(args.rules)
    W=W[W.size_cat.eq('small')].merge(sm[['PWSID','src']],on='PWSID')
    eta,lr=sp.truth_model(sm)
    source_files=[args.frame,args.rules,args.frozen_reference,Path(sp.__file__),Path(tl.__file__),Path(ac.__file__)]
    input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    records=[];replicate_records=[];checks=[];pseudo=[]
    start=time.perf_counter()
    for scenario in ['fitted','auc-match']:
        for seed in range(args.seeds):
            rng=np.random.default_rng(900+10*seed+list(sp.SCENARIOS).index(scenario))
            lin,scale=sp.scenario_linear(sm,eta,lr,*sp.SCENARIOS[scenario],rng)
            y=(rng.random(len(sm))<expit(lin)).astype(float)
            costs={c:np.zeros(len(sm)) for c in tl.RULE_COLS}
            for (yy,src),g in W.groupby(['y','src']):
                idx=np.flatnonzero((y==yy)&(sm.src.to_numpy()==src))
                pick=g.iloc[rng.integers(0,len(g),len(idx))]
                for c in tl.RULE_COLS:costs[c][idx]=pick[c].to_numpy()
            truth={'systems':float(y.mean()),'people':float(np.sum(sm.P.to_numpy()*y)/sm.P.sum())}
            for target in truth:
                old=frozen[(frozen.scenario==scenario)&(frozen.seed==seed)&(frozen.estimand==target)]
                assert np.allclose(old.truth,truth[target],atol=2e-14),(scenario,seed,target,truth[target],old.truth.to_list())
            pseudo.append({'scenario':scenario,'seed':seed,'generation_seed':900+10*seed+list(sp.SCENARIOS).index(scenario),'slope_scale':float(scale),'truth':truth,'y_sha256':array_hash(y),'donor_cost_hashes':{k:array_hash(v) for k,v in costs.items()},'truth_matches_frozen':True})

            # Exact original cell/reference mechanism and budget.
            cells=sm.cell.astype('category').cat.codes.to_numpy()
            obs_nh=np.bincount(cells[sm.y_mcl.notna().to_numpy()],minlength=cells.max()+1).astype(float)
            Nh=np.bincount(cells,minlength=len(obs_nh)).astype(float)
            ref_budget=800*np.sum(obs_nh/obs_nh.sum()*np.array([costs['cost_full'][cells==h].mean() for h in range(len(obs_nh))]))
            ref_pi=tl.budget_probs((obs_nh/obs_nh.sum()/Nh)[cells],costs['cost_full'],ref_budget)
            target_arrays={'systems':np.ones(len(sm)),'people':sm.P.to_numpy(float)}
            reference={}
            for target_index,(target,a) in enumerate(target_arrays.items()):
                # 5 non-dual strategies x 2 budgets before the reference; after
                # systems reference, another 10 non-dual calls precede people ref.
                rr=clone_rng(rng);rr.bit_generator.advance((10+11*target_index)*OLD_R*2*len(sm))
                rows=[];A=a.sum();sumAF=np.sum(a*sm.f.to_numpy())
                for k in range(R):
                    I=rr.random(len(sm))<ref_pi;J=rr.random(len(sm))<0
                    res=tl.estimate(y,y,sm.f.to_numpy(),a,ref_pi,I,J,0,A,sumAF,False)
                    cost=float(costs['cost_full'][I].sum());rows.append((*res,cost))
                first=summarize(rows[:OLD_R],truth[target])
                old=frozen[(frozen.scenario==scenario)&(frozen.seed==seed)&(frozen.estimand==target)&frozen.strategy.str.startswith('REFERENCE')].iloc[0]
                comparison={'scenario':scenario,'seed':seed,'estimand':target,'mse_first600':first['mse'],'mse_frozen':float(old.mse),'coverage_first600':first['cov_kg'],'coverage_frozen':float(old.cov_kg),'cost_first600':first['mean_cost'],'cost_frozen':float(old.cost_mean),'mse_identical':bool(np.isclose(first['mse'],old.mse,rtol=1e-10,atol=1e-14)),'coverage_identical':bool(np.isclose(first['cov_kg'],old.cov_kg,atol=1e-12)),'cost_identical':bool(np.isclose(first['mean_cost'],old.cost_mean,rtol=1e-12))}
                checks.append(comparison)
                assert comparison['mse_identical'] and comparison['coverage_identical'] and comparison['cost_identical'],comparison
                out=summarize(rows,truth[target]);reference[target]=out
                records.append({'scenario':scenario,'seed':seed,'estimand':target,'budget':ref_budget,'strategy':'Reference cell allocation; full schedules','expected_cost':float(np.sum(ref_pi*costs['cost_full'])),'mse_ratio_to_reference':1.0,**out})

            u=sm.u.to_numpy(float);P=sm.P.to_numpy(float);ff=sm.f.to_numpy(float)
            score=.5*(.25*u/u.mean()+.75*P*u/(P*u).mean())+.5*(.25+.75*P/P.mean())
            unit_cost=.2*costs['cost_full']+.8*costs['cost_R12']
            for budget in BUDGETS:
                pi=tl.budget_probs(score,unit_cost,budget)
                rr=np.random.default_rng(20261031+100000*seed+10000*list(sp.SCENARIOS).index(scenario)+budget)
                allrows={target:[] for target in target_arrays}
                for k in range(R):
                    I=rr.random(len(sm))<pi;J=rr.random(len(sm))<.2
                    actual_cost=float(np.sum(np.where(J[I],costs['cost_full'][I],costs['cost_R12'][I])))
                    for target,a in target_arrays.items():
                        res=tl.estimate(y,costs['yt_R12'],ff,a,pi,I,J,.2,a.sum(),np.sum(a*ff),True)
                        allrows[target].append((*res,actual_cost))
                for target,rows in allrows.items():
                    out=summarize(rows,truth[target]);ratio=out['mse']/reference[target]['mse']
                    ratio_mc_se=ratio*np.sqrt((out['mse_mc_se']/out['mse'])**2+(reference[target]['mse_mc_se']/reference[target]['mse'])**2)
                    records.append({'scenario':scenario,'seed':seed,'estimand':target,'budget':budget,'strategy':'Shared dual; R12; q=0.2','expected_cost':float(np.sum(pi*unit_cost)),'mse_ratio_to_reference':ratio,'mse_ratio_mc_se':ratio_mc_se,**out})
                    x=np.asarray(rows,float);lo,hi=ac.intervals_vec(x[:,0],x[:,1],x[:,2],x[:,3])['kg']
                    replicate_records.extend({'scenario':scenario,'seed':seed,'estimand':target,'budget':budget,'draw':k,'estimate':row[0],'se':row[1],'lo':lo[k],'hi':hi[k],'cost':row[4],'truth':truth[target]} for k,row in enumerate(x))
            pd.DataFrame(records).to_csv(OUT/'direct_budget_by_seed.csv',index=False)
            print(scenario,seed,'complete',round(time.perf_counter()-start,1),'seconds',flush=True)

    data=pd.DataFrame(records)
    agg=data.groupby(['scenario','estimand','budget','strategy'],sort=False).agg(mean_mse=('mse','mean'),mean_mse_ratio=('mse_ratio_to_reference','mean'),min_mse_ratio=('mse_ratio_to_reference','min'),max_mse_ratio=('mse_ratio_to_reference','max'),mean_coverage=('cov_kg','mean'),min_coverage=('cov_kg','min'),max_coverage=('cov_kg','max'),mean_interval_width=('mean_interval_width','mean'),mean_cost=('mean_cost','mean'),mean_n=('mean_n','mean')).reset_index()
    agg.to_csv(OUT/'direct_budget_summary.csv',index=False)
    pd.DataFrame(replicate_records).to_csv(OUT/'direct_budget_draws.csv',index=False)
    for p in source_files:assert hashlib.sha256(p.read_bytes()).hexdigest()==input_hashes[p.name]
    info={'pass':True,'source_sha256':input_hashes,'R':R,'OLD_R':OLD_R,'budgets':BUDGETS,'frozen_reference_reproduction_checks':checks,'pseudo_populations':pseudo,'pseudo_truth_matches_all_requested_frozen_settings':True,'original_reference_draws_exactly_reproduced_in_all_requested_targets':True,'same_I_J_used_for_both_dual_estimands':True,'LightGBM_retraining':False,'new_independent_validation':False,'elapsed_seconds':time.perf_counter()-start,'interpretation':'Conditional direct-budget re-evaluation in the original fitted/AUC-matched pseudo-populations; known design probabilities within simulation do not identify real-world historical selection.'}
    (OUT/'direct_budget_validation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(agg.to_string(index=False));print('ALL SOURCE FILES UNCHANGED',flush=True)

if __name__=='__main__':
    main()
