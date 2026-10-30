"""Low-cost diagnostics of the frozen small-system frame; no model retraining.

Alternative calibration and selection-odds scenarios are sensitivity analyses,
not estimates of unavailable official inclusion probabilities.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit,logit

import argparse
from project_paths import D, R

def lam(y,fs,a,w):
    yb=np.sum(w*a*y)/np.sum(w*a);fb=np.sum(w*a*fs)/np.sum(w*a)
    return np.clip(np.sum(w*a*a*(y-yb)*(fs-fb))/max(np.sum(w*a*a*(fs-fb)**2),1e-12),0,2)

def estimate(frame,weight,mode='tuned'):
    observed=frame.y_mcl.notna().to_numpy();ys=frame.loc[observed,'y_mcl'].to_numpy(float)
    fs=frame.f.to_numpy(float);result={}
    for target in ['systems','people']:
        a=np.ones(len(frame)) if target=='systems' else frame.P.to_numpy(float)
        aa=a[observed];ff=fs[observed]
        ll=lam(ys,ff,aa,weight) if mode=='tuned' else float(mode)
        theta=(ll*np.sum(a*fs)+np.sum(weight*aa*(ys-ll*ff)))/a.sum()
        hajek=np.sum(weight*aa*ys)/np.sum(weight*aa)
        result.update({f'{target}_difference':theta,f'{target}_hajek':hajek,f'{target}_lambda':ll})
    return result

def weights_for(frame,key):
    observed=frame.y_mcl.notna();counts=frame.groupby(key,observed=True).size()
    n=frame.loc[observed].groupby(key,observed=True).size()
    assert set(counts.index)<=set(n.index)
    return frame.loc[observed,key].map(counts/n).to_numpy(float)

def rake(frame,variables,initial,tol=1e-10,maxiter=10000):
    observed=frame.y_mcl.notna();sample=frame.loc[observed];w=initial.copy()
    margins={v:frame.groupby(v,observed=True).size() for v in variables}
    unsupported={v:list(set(margins[v].index)-set(sample[v].unique())) for v in variables}
    if any(unsupported.values()):return None,{'unsupported':unsupported}
    for it in range(maxiter):
        for v in variables:
            got=pd.Series(w,index=sample.index).groupby(sample[v],observed=True).sum()
            w*=sample[v].map(margins[v]/got).to_numpy(float)
        error=max(float(np.max(np.abs(pd.Series(w,index=sample.index).groupby(sample[v],observed=True).sum()/margins[v]-1))) for v in variables)
        if error<tol:break
    return w,{'iterations':it+1,'maximum_relative_margin_error':error,'unsupported':unsupported}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frame',type=Path,default=D/'small_frame_anchor.parquet')
    ap.add_argument('--occurrence',type=Path,default=D/'ucmr5_all.parquet')
    ap.add_argument('--output',type=Path,default=R/'review_sensitivity')
    ap.add_argument('--replicates',type=int,default=4000)
    ap.add_argument('--seed',type=int,default=20261031)
    args=ap.parse_args()
    if args.replicates < 2: ap.error('--replicates must be at least 2')
    OUT=args.output; OUT.mkdir(parents=True,exist_ok=True); SRC=args.frame
    d=pd.read_parquet(SRC);lab=d.y_mcl.notna().values;y=d.loc[lab,'y_mcl'].to_numpy(float)
    f=d.f.to_numpy(float);N=len(d);P=d.P.to_numpy(float)
    rows=[];notes={}
    for name,key in [('Unweighted frame expansion','one'),('Source only','src'),('Size only','scls'),('Source-by-size (primary)','cell')]:
        x=d.copy();x['one']='all';w=weights_for(x,key)
        out=estimate(x,w);rows.append(dict(scheme=name,n=len(w),N=len(x),weight_min=w.min(),weight_max=w.max(),kish_effective_n=w.sum()**2/np.sum(w*w),**out))
    basew=weights_for(d,'cell')
    for mode in [0,1]:
        out=estimate(d,basew,mode)
        rows.append(dict(scheme=f'Primary cells; lambda={mode}',n=len(basew),N=N,weight_min=basew.min(),weight_max=basew.max(),kish_effective_n=basew.sum()**2/np.sum(basew*basew),**out))
    for name,vars in [('Cell + EPA-region margins',['cell','EPA_REGION']),('Cell + primacy margins',['cell','PRIMACY_AGENCY_CODE'])]:
        w,info=rake(d,vars,basew);notes[name]=info
        if w is None:continue
        out=estimate(d,w)
        rows.append(dict(scheme=name,n=len(w),N=N,weight_min=w.min(),weight_max=w.max(),kish_effective_n=w.sum()**2/np.sum(w*w),**out))
    x=d[d['pop']<3300].copy();w=weights_for(x,'cell');out=estimate(x,w)
    rows.append(dict(scheme='Exclude exactly-3300 systems',n=len(w),N=len(x),weight_min=w.min(),weight_max=w.max(),kish_effective_n=w.sum()**2/np.sum(w*w),**out))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'weight_calibration_sensitivity.csv',index=False)

    cells=d.groupby('cell',observed=True).agg(N=('PWSID','size'),n=('y_mcl','count'),positives=('y_mcl','sum'),mean_f=('f','mean'))
    cells['observed_prevalence']=cells.positives/cells.n;cells['frame_weight']=cells.N/cells.n
    cells['frame_share']=cells.N/N;cells['weighted_positive_contribution']=cells.frame_share*cells.observed_prevalence
    cells['share_of_poststratified_variance']=(cells.N/N)**2*(1-cells.n/cells.N)*cells.observed_prevalence*(1-cells.observed_prevalence)/(cells.n-1)
    cells['share_of_poststratified_variance']/=cells.share_of_poststratified_variance.sum()
    cells.to_csv(OUT/'cell_influence.csv')

    # A transparent nonresponse/selection scenario: within each primary cell,
    # odds(unmeasured exceedance)/odds(measured exceedance) = gamma.
    # The gamma=1 baseline is the post-stratified design-only estimate.
    qh=cells.observed_prevalence.to_numpy();unobs=(cells.N-cells.n).to_numpy();positive=cells.positives.sum()
    def scenario(gamma):
        q=gamma*qh/(1-qh+gamma*qh)
        return (positive+np.sum(unobs*q))/N
    plugin=float(f.mean());tip=brentq(lambda g:scenario(g)-plugin,.01,10)
    scenarios=pd.DataFrame([{'gamma':g,'system_prevalence':scenario(g),'system_count':scenario(g)*N,'interpretation':'assumed odds among unmeasured relative to measured within source-by-size cells'} for g in [.25,.5,.75,1,1.25,1.5,2,4,tip]])
    scenarios.to_csv(OUT/'within_cell_selection_odds_scenarios.csv',index=False)
    pd.DataFrame([{'gamma':float(g),'log2_gamma':float(np.log2(g)),'system_prevalence':scenario(g),'system_count':scenario(g)*N} for g in np.geomspace(.25,4,81)]).to_csv(OUT/'within_cell_selection_odds_curve.csv',index=False)

    # Cell-stratified multinomial bootstrap of the two concentration ratios.
    # Retains the original estimator separately in each size subgroup and nationally.
    rng=np.random.default_rng(args.seed);B=args.replicates
    sample=d.loc[lab].reset_index(drop=True)
    cell_indices=[np.flatnonzero(sample.cell.eq(k).to_numpy()) for k in cells.index]
    counts=np.zeros((B,len(sample)),dtype=np.int16)
    for idx in cell_indices:counts[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=B)
    def vector_est(a,subset,return_lambda=False):
        sy=sample.y_mcl.to_numpy(float);sf=sample.f.to_numpy(float);asamp=a[lab]
        c=counts[:,subset[lab]].astype(float);yy=sy[subset[lab]];ff=sf[subset[lab]];aa=asamp[subset[lab]];ww=basew[subset[lab]]
        weighted=c*(ww*aa);yb=(weighted@yy)/weighted.sum(axis=1);fb=(weighted@ff)/weighted.sum(axis=1)
        z=c*(ww*aa*aa)
        numerator=np.sum(z*(yy-yb[:,None])*(ff-fb[:,None]),axis=1)
        denominator=np.sum(z*(ff-fb[:,None])**2,axis=1)
        ll=np.clip(numerator/np.maximum(denominator,1e-12),0,2)
        total=ll*np.sum(a[subset]*f[subset])+np.sum(c*(ww*aa)*(yy[None,:]-ll[:,None]*ff[None,:]),axis=1)
        return (total,ll) if return_lambda else total
    allmask=np.ones(N,dtype=bool);very=d['pop'].le(100).to_numpy()
    ns,lambda_system_draws=vector_est(np.ones(N),allmask,True);nv=vector_est(np.ones(N),very)
    ps,lambda_people_draws=vector_est(P,allmask,True);pv=vector_est(P,very)
    ratios={'very_small_share_of_affected_systems':nv/ns,'very_small_share_of_affected_people':pv/ps}
    concentration={k:{'bootstrap_median':float(np.median(v)),'percentile_025':float(np.quantile(v,.025)),'percentile_975':float(np.quantile(v,.975))} for k,v in ratios.items()}
    pd.DataFrame(ratios).to_csv(OUT/'concentration_stratified_bootstrap.csv',index=False)

    size_rows=[];size_total_draws={};size_point_totals={};common_total_draws={}
    for target,aa,denominator in [('systems',np.ones(N),ns),('people',P,ps)]:
        size_total_draws[target]={};size_point_totals[target]={};common_total_draws[target]={}
        national_point=estimate(d,basew)[f'{target}_difference']*aa.sum()
        national_lambda=estimate(d,basew)[f'{target}_lambda']
        for size in ['<=100','101-500','501-1000','1001-3300']:
            subset=d.scls.eq(size).to_numpy();g=d.loc[subset].copy()
            wg=weights_for(g,'cell');point=estimate(g,wg)[f'{target}_difference']*aa[subset].sum()
            totals=vector_est(aa,subset);shares=totals/denominator
            size_total_draws[target][size]=totals;size_point_totals[target][size]=point
            # Common national lambda gives additive domain totals, useful as a
            # separate composition diagnostic, not silently substituted for s19.
            sel=subset[lab];common=(national_lambda*np.sum(aa[subset]*f[subset])+np.sum(basew[sel]*aa[lab][sel]*(y[sel]-national_lambda*f[lab][sel])))
            # Additive domain analysis: apply the nationally estimated coefficient
            # to each domain; re-estimate that one national coefficient in every
            # bootstrap draw and use the same draw's national total denominator.
            ll=lambda_system_draws if target=='systems' else lambda_people_draws
            c=counts[:,sel].astype(float);ys=y[sel];ff=f[lab][sel];aas=aa[lab][sel];ww=basew[sel]
            common_draws=ll*np.sum(aa[subset]*f[subset])+np.sum(c*(ww*aas)*(ys[None,:]-ll[:,None]*ff[None,:]),axis=1)
            common_total_draws[target][size]=common_draws
            common_shares=common_draws/denominator
            size_rows.append({'size_band':size,'estimand':target,'point_affected_total':point,'affected_total_percentile025':float(np.quantile(totals,.025)),'affected_total_percentile975':float(np.quantile(totals,.975)),'national_total_point':national_point,'point_share_of_national_total':point/national_point,'share_percentile025':float(np.quantile(shares,.025)),'share_percentile975':float(np.quantile(shares,.975)),'common_national_lambda_point_total':common,'common_national_lambda_point_share':common/national_point,'common_national_lambda_share_percentile025':float(np.quantile(common_shares,.025)),'common_national_lambda_share_percentile975':float(np.quantile(common_shares,.975)),'common_national_lambda_total_percentile025':float(np.quantile(common_draws,.025)),'common_national_lambda_total_percentile975':float(np.quantile(common_draws,.975)),'bootstrap_replicates':B,'coefficient':'lambda re-estimated separately within every domain and nationally in every resample, matching frozen s19; common_national columns instead use one re-estimated national lambda per target/draw','denominator':'separately estimated national total; original domain totals need not sum exactly to national total; common_national domain totals are additive'})
    sizes=pd.DataFrame(size_rows)
    for target in ['systems','people']:
        normalization=np.sum(list(size_total_draws[target].values()),axis=0)
        pointnorm=sum(size_point_totals[target].values())
        for size in ['<=100','101-500','501-1000','1001-3300']:
            mask=(sizes.estimand==target)&(sizes.size_band==size)
            composition=size_total_draws[target][size]/normalization
            sizes.loc[mask,'point_normalized_domain_share']=size_point_totals[target][size]/pointnorm
            sizes.loc[mask,'normalized_domain_share_percentile025']=np.quantile(composition,.025)
            sizes.loc[mask,'normalized_domain_share_percentile975']=np.quantile(composition,.975)
    sizes.to_csv(OUT/'size_band_burden_bootstrap.csv',index=False)
    canonical=sizes[['size_band','estimand','common_national_lambda_point_total','common_national_lambda_total_percentile025','common_national_lambda_total_percentile975','common_national_lambda_point_share','common_national_lambda_share_percentile025','common_national_lambda_share_percentile975']].copy()
    canonical.columns=['size_band','estimand','affected_total','affected_total_lo','affected_total_hi','affected_share','affected_share_lo','affected_share_hi']
    canonical['frame_target_total']=[float(np.sum(np.ones(N)[d.scls.eq(k).to_numpy()])) if t=='systems' else float(P[d.scls.eq(k).to_numpy()].sum()) for k,t in zip(canonical.size_band,canonical.estimand)]
    for s in ['','_lo','_hi']:
        canonical['prevalence'+s]=canonical['affected_total'+s]/canonical.frame_target_total
        canonical['prevalence'+s+'_pct']=100*canonical['prevalence'+s]
        canonical['affected_share'+s+'_pct']=100*canonical['affected_share'+s]
    canonical['coefficient']='one nationally re-estimated lambda per target/bootstrap draw; same coefficient across size domains'
    canonical['interval']='2.5th/97.5th percentiles of 4000 source-by-size-cell stratified bootstrap resamples, f and frame fixed'
    canonical.to_csv(OUT/'additive_domain_burden_canonical.csv',index=False)
    inventory=canonical[['size_band','estimand','frame_target_total']].copy()
    inventory['inventory_share_pct']=inventory['frame_target_total']/inventory.groupby('estimand').frame_target_total.transform('sum')*100
    inventory.rename(columns={'size_band':'class'}).to_csv(OUT/'fig5c_full_inventory_shares.csv',index=False)
    pd.DataFrame([{'size_band':k,'affected_systems':f"{canonical[(canonical.size_band==k)&(canonical.estimand=='systems')].affected_total.iloc[0]:,.0f}",
                   'associated_people_thousands':f"{canonical[(canonical.size_band==k)&(canonical.estimand=='people')].affected_total.iloc[0]/1000:,.1f}",
                   'system_share_percent_interval':f"{canonical[(canonical.size_band==k)&(canonical.estimand=='systems')].affected_share_pct.iloc[0]:.1f} ({canonical[(canonical.size_band==k)&(canonical.estimand=='systems')].affected_share_lo_pct.iloc[0]:.1f}–{canonical[(canonical.size_band==k)&(canonical.estimand=='systems')].affected_share_hi_pct.iloc[0]:.1f})",
                   'population_share_percent_interval':f"{canonical[(canonical.size_band==k)&(canonical.estimand=='people')].affected_share_pct.iloc[0]:.1f} ({canonical[(canonical.size_band==k)&(canonical.estimand=='people')].affected_share_lo_pct.iloc[0]:.1f}–{canonical[(canonical.size_band==k)&(canonical.estimand=='people')].affected_share_hi_pct.iloc[0]:.1f})"} for k in ['<=100','101-500','501-1000','1001-3300']]).to_csv(OUT/'tableS9_additive_domains_display.csv',index=False)
    for target in ['systems','people']:
        assert np.isclose(sizes.loc[sizes.estimand.eq(target),'common_national_lambda_point_share'].sum(),1)
        national_draws=ns if target=='systems' else ps
        assert np.allclose(np.sum(list(common_total_draws[target].values()),axis=0),national_draws,rtol=1e-12,atol=1e-7)
    fig5c=sizes[['size_band','estimand','common_national_lambda_point_share','common_national_lambda_share_percentile025','common_national_lambda_share_percentile975']].copy()
    fig5c.columns=['class','estimand','share','lo','hi']
    for col in ['share','lo','hi']:fig5c[col+'_pct']=fig5c[col]*100
    fig5c.to_csv(OUT/'fig5c_additive_domain_shares.csv',index=False)
    paired=common_total_draws['systems']['<=100']/ns-common_total_draws['people']['<=100']/ps
    point_paired=canonical.loc[(canonical.size_band=='<=100')&(canonical.estimand=='systems'),'affected_share'].iloc[0]-canonical.loc[(canonical.size_band=='<=100')&(canonical.estimand=='people'),'affected_share'].iloc[0]
    paired_info={'point_difference_percentage_points':float(point_paired*100),'percentile025_percentage_points':float(np.quantile(paired,.025)*100),'percentile975_percentage_points':float(np.quantile(paired,.975)*100),'bootstrap_fraction_positive':float(np.mean(paired>0)),'definition':'common-lambda share of estimated affected systems minus share of associated service population for <=100 people systems; paired resampling across targets'}
    (OUT/'very_small_paired_share_contrast.json').write_text(json.dumps(paired_info,indent=2),encoding='utf-8')

    raw=pd.read_parquet(args.occurrence,columns=['PWSID','Size']).drop_duplicates()
    cross=d.loc[lab,['PWSID','pop','y_mcl','cell']].merge(raw,on='PWSID',validate='one_to_one')
    cross.to_csv(OUT/'current_small_vs_ucmr_size_crosswalk.csv',index=False)
    info={'source':SRC.name,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'observed_systems':int(lab.sum()),'observed_positive_systems':int(y.sum()),'unweighted_observed_prevalence':float(y.mean()),'exactly_3300_frame':int(d['pop'].eq(3300).sum()),'exactly_3300_observed':int((d['pop'].eq(3300)&d.y_mcl.notna()).sum()),'original_ucmr_size_among_current_small':cross.Size.value_counts().to_dict(),'original_ucmr_L_among_current_small_positives':int(cross.loc[cross.Size.eq('L'),'y_mcl'].sum()),'original_prediction_mean':plugin,'selection_odds_gamma_at_plugin':float(tip),'raking_diagnostics':notes,'legacy_independently_fitted_domain_concentration_bootstrap':concentration,'canonical_domain_estimator':'common nationally estimated lambda per target, re-estimated in every resample and then applied to all four size domains','canonical_domain_additivity_each_of_4000_draws_verified':True,'canonical_domain_csv':'additive_domain_burden_canonical.csv','bootstrap_replicates':B,'bootstrap_seed':args.seed,'interpretation':'Sensitivity calculations conditional on the frozen frame/model and assumed source-by-size exchangeability; not recovered official selection probabilities and not a prospective validation.'}
    (OUT/'sensitivity_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(info,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
