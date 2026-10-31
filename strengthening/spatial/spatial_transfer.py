from strengthening import paths
"""Fixed-parameter, full-pipeline geographic transfer evaluation.

Larger-system outcomes are the only supervision. For each held EPA region,
classifier training and neighbour-label reference pools exclude that region.
Training neighbour predictors also exclude each row's own region, matching
the cold-region information regime. No held-region outcome enters fitting,
neighbour construction, category fitting, calibration or model selection.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score,average_precision_score

BASE=['lat','lon','sw','purch','cws','owner','school','whole','lpop','lconn','state','region','age',
      'n_wells','n_intakes','n_springs','n_consec','n_sources','n_treat_plants','n_sellers']
ATTR=[x for x in BASE if x not in ['lat','lon','state','region']]
NB=['nb_n10','nb_mcl10','nb_det10','nb_n25','nb_mcl25','nb_det25','nb_n50','nb_mcl50','nb_det50','nb_kmcl','nb_kdet','nb_dmin']
SETS={'attributes':ATTR,'geography':BASE,'neighbours':BASE+NB}
GB=dict(n_estimators=500,learning_rate=.02,num_leaves=15,min_child_samples=30,subsample=.8,subsample_freq=1,
        colsample_bytree=.7,verbosity=-1,n_jobs=2)

def setup(frame):
    f=frame.copy()
    f['block']=pd.to_numeric(f.EPA_REGION).astype(int)
    f['sw']=f.GW_SW_CODE.eq('SW').astype(int)
    f['purch']=f.PRIMARY_SOURCE_CODE.fillna('').str.endswith('P').astype(int)
    f['cws']=f.PWS_TYPE_CODE.eq('CWS').astype(int)
    f['school']=f.IS_SCHOOL_OR_DAYCARE_IND.eq('Y').astype(int)
    f['whole']=f.IS_WHOLESALER_IND.eq('Y').astype(int)
    f['owner']=f.OWNER_TYPE_CODE
    f['state']=f.PRIMACY_AGENCY_CODE
    f['region']=f.EPA_REGION
    f['lpop']=np.log10(f['pop'].clip(lower=1))
    f['lconn']=np.log10(f.conn.clip(lower=1))
    f['age']=2026-pd.to_datetime(f.FIRST_REPORTED_DATE,format='%m/%d/%Y',errors='coerce').dt.year
    return f

def nb_features(target,reference):
    out=np.full((len(target),len(NB)),np.nan)
    valid=target[['lat','lon']].notna().all(axis=1).to_numpy()
    ref=reference[reference[['lat','lon']].notna().all(axis=1)]
    if not len(ref):return pd.DataFrame(out,columns=NB,index=target.index)
    tree=BallTree(np.radians(ref[['lat','lon']].to_numpy()),metric='haversine')
    idx,dist=tree.query_radius(np.radians(target.loc[valid,['lat','lon']].to_numpy()),r=50/6371.,return_distance=True)
    rid=ref.PWSID.to_numpy();y=ref.y_mcl.to_numpy();d=ref.y_det.to_numpy()
    for row,(ii,dd) in zip(np.flatnonzero(valid),zip(idx,dist)):
        keep=rid[ii]!=target.PWSID.iloc[row];ii=ii[keep];dd=dd[keep]*6371.
        for j,radius in enumerate([10,25,50]):
            m=dd<=radius;out[row,3*j]=m.sum()
            if m.any():out[row,3*j+1]=y[ii[m]].mean();out[row,3*j+2]=d[ii[m]].mean()
        if len(dd):
            w=np.exp(-dd/10);out[row,9]=np.average(y[ii],weights=w);out[row,10]=np.average(d[ii],weights=w);out[row,11]=dd.min()
    return pd.DataFrame(out,columns=NB,index=target.index)

def training_nb(train):
    out=pd.DataFrame(np.nan,index=train.index,columns=NB)
    for block,rows in train.groupby('block'):
        ref=train[train.block!=block]
        out.loc[rows.index]=nb_features(rows,ref)
    return out

def category_maps(train,test):
    train=train.copy();test=test.copy();mapping={}
    for key in ['owner','state','region']:
        cats=sorted(train[key].dropna().astype(str).unique())
        mapping[key]=cats
        train[key]=pd.Categorical(train[key].astype('string'),categories=cats)
        test[key]=pd.Categorical(test[key].astype('string'),categories=cats)
    return train,test,mapping

def source_buffer_exclusion(train,target,km):
    q=target[target[['lat','lon']].notna().all(axis=1)&target.y_mcl.notna()]
    if not len(q):return np.zeros(len(train),bool)
    tree=BallTree(np.radians(q[['lat','lon']].to_numpy()),metric='haversine')
    valid=train[['lat','lon']].notna().all(axis=1).to_numpy();mask=np.zeros(len(train),bool)
    counts=tree.query_radius(np.radians(train.loc[valid,['lat','lon']].to_numpy()),r=km/6371.,count_only=True)
    mask[valid]=counts>0
    return mask

def run(frame_path,out,seeds=(0,1,2),buffer_km=0,source_complete_coordinates=False):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    f=setup(pd.read_parquet(frame_path));source=f[f.size_cat.isin(['medium','large'])&f.y_mcl.notna()].copy()
    assert len(f)==66395 and len(source)==9229
    inputs={'frame_sha256':hashlib.sha256(frame_path.read_bytes()).hexdigest(),'frame_n':len(f),'larger_source_n':len(source),
            'small_observed_n':int(((f.size_cat=='small')&f.y_mcl.notna()).sum()),'seeds':list(seeds),'parameters':GB,
            'heldout_group':'EPA region','buffer_km':buffer_km,'source_complete_coordinates':source_complete_coordinates or bool(buffer_km),'feature_sets':SETS,
            'category_policy':'Training-only nominal categories; unseen test categories are missing. No target fitted calibration.',
            'training_neighbours':'References omit outer held region and each training row own region.',
            'endpoint':'Any retained-location mean PFOA or PFOS >=4.05ng/L; belowMRLzero; existing fixed endpoint.'}
    (out/'analysis_specification.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    records=[];audit=[]
    for block in range(1,11):
        fold=out/f'fold_{block:02d}.parquet';audit_path=out/f'fold_{block:02d}_audit.json'
        if fold.exists() and audit_path.exists():
            records.append(pd.read_parquet(fold));audit.append(json.loads(audit_path.read_text()));continue
        test=f[f.block==block].copy();train=source[source.block!=block].copy()
        before=len(train)
        if buffer_km or source_complete_coordinates:
            train=train[train[['lat','lon']].notna().all(axis=1)].copy()
        if buffer_km:
            train=train.loc[~source_buffer_exclusion(train,test,buffer_km)].copy()
        assert not train.PWSID.isin(test.PWSID).any()
        assert block not in train.block.unique()
        train=train.join(training_nb(train));test=test.join(nb_features(test,train))
        train,test,cats=category_maps(train,test)
        cat_audit={k:{'training_categories':v,'unseen_test_count':int(test[k].isna().sum())} for k,v in cats.items()}
        pred=test[['PWSID','block','PRIMACY_AGENCY_CODE','STATE_CODE','size_cat','pop','lat','lon','y_mcl','y_det']].copy()
        pred['constant_systems']=train.y_mcl.mean()
        pred['constant_people']=np.average(train.y_mcl,weights=train['pop'].fillna(25).clip(lower=25))
        for name,cols in SETS.items():
            if buffer_km and name!='neighbours':continue
            for seed in seeds:
                model=lgb.LGBMClassifier(**GB,random_state=seed)
                model.fit(train[cols],train.y_mcl.astype(int))
                pred[f'{name}_seed{seed}']=model.predict_proba(test[cols])[:,1]
            pred[name]=pred[[f'{name}_seed{s}' for s in seeds]].mean(axis=1)
        current={'held_region':block,'train_n_before_buffer':before,'train_n':len(train),'train_positive':int(train.y_mcl.sum()),
                 'source_regions':sorted(train.block.unique().tolist()),'train_test_id_overlap':0,'held_region_reference_n':0,
                 'test_frame_n':len(test),'test_observed_n':int(test.y_mcl.notna().sum()),'test_small_observed_n':int(((test.size_cat=='small')&test.y_mcl.notna()).sum()),
                 'target_has_neighbour_count':int(test.nb_n50.fillna(0).gt(0).sum()),
                 'observed_target_has_neighbour_count':int((test.nb_n50.fillna(0).gt(0)&test.y_mcl.notna()).sum()),
                 'source_ids_sha256':hashlib.sha256('\n'.join(sorted(train.PWSID)).encode()).hexdigest(),'category_audit':cat_audit,
                 'elapsed_seconds':time.time()-start}
        pred.to_parquet(fold,index=False);audit_path.write_text(json.dumps(current,indent=2),encoding='utf-8')
        records.append(pred);audit.append(current)
        print(f'Completed region {block};train={len(train)},observedtest={current["test_observed_n"]},elapsed={time.time()-start:.1f}s',flush=True)
    p=pd.concat(records,ignore_index=True).sort_values('PWSID').reset_index(drop=True)
    assert len(p)==len(f) and p.PWSID.nunique()==len(f)
    p.to_parquet(out/'spatial_predictions_frame.parquet',index=False)
    p.to_csv(out/'spatial_predictions_frame.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='category_audit'} for r in audit]).to_csv(out/'spatial_fold_audit.csv',index=False)
    inputs['elapsed_seconds']=time.time()-start;inputs['all_fold_audits_passed']=True
    inputs['unchanged_input_sha256']=hashlib.sha256(frame_path.read_bytes()).hexdigest()==inputs['frame_sha256']
    (out/'analysis_completion.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    return p

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--frame',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--seeds',default='0,1,2');a.add_argument('--buffer-km',type=float,default=0);a.add_argument('--source-complete-coordinates',action='store_true')
    args=a.parse_args();run(args.frame,args.out,tuple(map(int,args.seeds.split(','))),args.buffer_km,args.source_complete_coordinates)
