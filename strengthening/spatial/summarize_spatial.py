from strengthening import paths
"""Conditional, paired EPA-region bootstrap for fixed out-of-region predictors."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

def weighted_u(y,p,w,g):
    u=np.zeros((10,10));pos=np.bincount(g,weights=w*y,minlength=10);neg=np.bincount(g,weights=w*(1-y),minlength=10)
    for r in range(10):
        pp=p[(g==r)&(y==1)];pw=w[(g==r)&(y==1)]
        for s in range(10):
            nn=p[(g==s)&(y==0)];nw=w[(g==s)&(y==0)]
            order=np.argsort(nn);nn=nn[order];cs=np.r_[0,np.cumsum(nw[order])]
            left=np.searchsorted(nn,pp,side='left');right=np.searchsorted(nn,pp,side='right')
            u[r,s]=np.sum(pw*(cs[left]+.5*(cs[right]-cs[left])))
    return u,pos,neg

def metrics(y,p,w,g,C):
    u,pos,neg=weighted_u(y,p,w,g)
    pden=(C@pos)*(C@neg);pooled=np.einsum('bi,ij,bj->b',C,u,C)/np.where(pden>0,pden,np.nan)
    within_den=C@(pos*neg);within=(C@np.diag(u))/np.where(within_den>0,within_den,np.nan)
    valid=(pos*neg)>0;local=np.diag(u)[valid]/(pos*neg)[valid]
    macro=(C[:,valid]@local)/np.where(C[:,valid].sum(axis=1)>0,C[:,valid].sum(axis=1),np.nan)
    sw=np.bincount(g,weights=w,minlength=10);sy=np.bincount(g,weights=w*y,minlength=10);sp=np.bincount(g,weights=w*p,minlength=10)
    sl=np.bincount(g,weights=w*(y-p)**2,minlength=10)
    pp=np.clip(p,1e-9,1-1e-9);ll=np.bincount(g,weights=w*(-y*np.log(pp)-(1-y)*np.log1p(-pp)),minlength=10)
    denom=C@sw
    values={'auc':pooled,'within_region_auc':within,'region_macro_auc':macro,'brier':(C@sl)/denom,'log_loss':(C@ll)/denom,
            'observed_prevalence':(C@sy)/denom,'mean_prediction':(C@sp)/denom,'calibration_bias':(C@(sp-sy))/denom}
    assert abs(pooled[0]-roc_auc_score(y,p,sample_weight=w))<1e-10
    point={k:v[0] for k,v in values.items()};point['pr_auc']=average_precision_score(y,p,sample_weight=w)
    point.update(n=len(y),positives=int(y.sum()),n_auc_regions=int(valid.sum()),weight_sum=w.sum())
    for k,v in values.items():
        lo,hi=np.nanpercentile(v[1:],[2.5,97.5]);point[k+'_lo']=lo;point[k+'_hi']=hi
    return point,values

def summary(path,frame_path,out,boot=2000,geo_only=False):
    out.mkdir(exist_ok=True,parents=True);data=pd.read_parquet(path)
    frame=pd.read_parquet(frame_path);small=frame[frame.size_cat=='small'].copy()
    small['cell']=small.GW_SW_CODE.eq('SW').astype(str)+'|'+pd.cut(small['pop'],[-1,100,500,1000,3300],labels=['a','b','c','d']).astype(str)
    n=small[small.y_mcl.notna()].groupby('cell').size();N=small.groupby('cell').size()
    W=small.PWSID.to_frame().assign(working_weight=small.cell.map(N/n))
    data=data.merge(W,on='PWSID',how='left');obs=data[data.y_mcl.notna()].copy()
    rng=np.random.default_rng(20261031);C=np.vstack([np.ones(10),rng.multinomial(10,np.full(10,.1),size=boot)])
    rows=[];contrasts=[];region_rows=[];seed_rows=[];modelnames=[m for m in ['attributes','geography','neighbours','coordinates','coordinates_neighbours'] if m in obs.columns]
    groups={'all_observed':obs,'small_observed':obs[obs.size_cat=='small'],'medium_observed':obs[obs.size_cat=='medium'],'large_observed':obs[obs.size_cat=='large']}
    if geo_only:groups={k:v[v.lat.notna()&v.lon.notna()] for k,v in groups.items()}
    for group,d in groups.items():
        y=d.y_mcl.to_numpy().astype(int);g=d.block.to_numpy().astype(int)-1
        weightings={'systems':np.ones(len(d)),'people':d['pop'].fillna(25).clip(lower=25).to_numpy()}
        if group=='small_observed':
            weightings['working_poststrat_systems']=d.working_weight.to_numpy()
            weightings['working_poststrat_people']=d.working_weight.to_numpy()*d['pop'].clip(lower=25).to_numpy()
        for weighting,w in weightings.items():
            stored={}
            for model in ['training_prevalence']+modelnames:
                p=d['constant_people' if weighting.endswith('people') else 'constant_systems'].to_numpy() if model=='training_prevalence' else d[model].to_numpy()
                point,vals=metrics(y,p,w,g,C);stored[model]=vals
                rows.append(dict(cohort=group,weighting=weighting,model=model,**point))
                if weighting=='systems':
                    for r in range(10):
                        m=g==r;yy=y[m];ff=p[m]
                        region_rows.append(dict(cohort=group,region=r+1,model=model,n=int(m.sum()),positives=int(yy.sum()),
                                                observed_prevalence=yy.mean() if len(yy) else np.nan,mean_prediction=ff.mean() if len(yy) else np.nan,
                                                auc=roc_auc_score(yy,ff) if len(np.unique(yy))==2 else np.nan,brier=np.mean((yy-ff)**2) if len(yy) else np.nan))
            for numerator,denominator in [('neighbours','geography'),('coordinates_neighbours','coordinates'),('coordinates','geography')]:
                if numerator not in stored or denominator not in stored:continue
                for metric in ['auc','within_region_auc','region_macro_auc','brier','calibration_bias']:
                    delta=stored[numerator][metric]-stored[denominator][metric]
                    contrasts.append(dict(cohort=group,weighting=weighting,comparison=numerator+'_minus_'+denominator,metric=metric,
                                          difference=delta[0],lo=np.nanpercentile(delta[1:],2.5),hi=np.nanpercentile(delta[1:],97.5)))
        for model in modelnames:
            for seed in [0,1,2]:
                col=f'{model}_seed{seed}'
                if col in d:
                    p=d[col].to_numpy();seed_rows.append(dict(cohort=group,model=model,seed=seed,n=len(y),positives=int(y.sum()),auc=roc_auc_score(y,p),brier=np.mean((y-p)**2)))
    pd.DataFrame(rows).to_csv(out/'spatial_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'spatial_paired_contrasts.csv',index=False)
    pd.DataFrame(region_rows).to_csv(out/'spatial_region_metrics.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'spatial_seed_metrics.csv',index=False)
    roc=[]
    from sklearn.metrics import roc_curve
    for group,d in groups.items():
        for model in modelnames:
            x,y,t=roc_curve(d.y_mcl,d[model]);roc.extend(dict(cohort=group,model=model,fpr=a,tpr=b,threshold=c) for a,b,c in zip(x,y,t))
    pd.DataFrame(roc).to_csv(out/'spatial_roc.csv',index=False)
    (out/'summary_specification.json').write_text(json.dumps({'bootstrap':'2000pairedEPA-regioncluster resamples; predictionsfixed; 10regions. Percentilesareconditional model-performanceuncertainty,not populationburdenCIs.','within_region_auc':'Concordantpair count pooled onlywithin heldEPAregions; macro gives each region with bothclasses equalweight.','working_weights':'Forobservedsmallcohort only, eightcurrent source-by-sizecell expansion weights assumeexchangeability; not originalEPAinclusionprobabilities.','buffer_metrics':'50km sensitivity evaluatedonlywhere targetcoordinatesknown.'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("cohort=='small_observed' and weighting=='systems'")[['model','n','positives','auc','auc_lo','auc_hi','within_region_auc','within_region_auc_lo','within_region_auc_hi','brier','calibration_bias']].to_string(index=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--predictions',type=Path,required=True);a.add_argument('--frame',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--geo-only',action='store_true');args=a.parse_args();summary(args.predictions,args.frame,args.out,geo_only=args.geo_only)
