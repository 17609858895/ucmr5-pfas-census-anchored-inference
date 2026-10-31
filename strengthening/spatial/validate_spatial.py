from strengthening import paths
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd,lightgbm,sklearn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from . import spatial_transfer as st
from .summarize_spatial import weighted_u
P=paths.SPATIAL_RESULTS;R=paths.WORKDIR
F=st.setup(pd.read_parquet(paths.DERIVED/'frame_systems.parquet'))
source=F[F.size_cat.isin(['medium','large'])&F.y_mcl.notna()]
tests={}
# A target's outcome is never inspected by the neighbour builder.
ref=pd.DataFrame({'PWSID':['a','b'],'lat':[0.,.01],'lon':[0.,0.],'y_mcl':[0.,1.],'y_det':[0.,1.],'block':[1,2]})
tar=pd.DataFrame({'PWSID':['t'],'lat':[.001],'lon':[0.],'y_mcl':[0.],'y_det':[0.],'block':[3]})
a=st.nb_features(tar,ref);b=st.nb_features(tar.assign(y_mcl=1.,y_det=1.),ref)
assert a.equals(b);tests['target_label_perturbation_invariance']=True
own=st.nb_features(ref.iloc[[0]],ref);assert own.nb_n50.iloc[0]==1 and own.nb_mcl50.iloc[0]==1;tests['own_pwsid_neighbour_exclusion']=True
tr=pd.DataFrame({'owner':['P','R'],'state':['AA','BB'],'region':['1','2']});te=pd.DataFrame({'owner':['P'],'state':['ZZ'],'region':['9']})
_,mapped,_=st.category_maps(tr,te);assert mapped.state.isna().all() and mapped.region.isna().all();tests['unseen_categories_not_fitted_from_target']=True
# A source row's own region never enters its label-neighbour representation.
a=st.training_nb(ref);r2=ref.copy();r2.loc[r2.block==1,['y_mcl','y_det']]=1
b=st.training_nb(r2);assert a.loc[ref.block==1].equals(b.loc[ref.block==1]);tests['own_region_label_perturbation_invariance']=True
# Weighted concordance matrix reproduces AUC under resampled region weights.
rng=np.random.default_rng(20261031);y=rng.integers(0,2,100);p=rng.random(100);w=rng.random(100)+.1;g=rng.integers(0,10,100)
u,pos,neg=weighted_u(y,p,w,g)
for _ in range(20):
    c=rng.multinomial(10,np.full(10,.1));actual=roc_auc_score(y,p,sample_weight=w*c[g]);expect=(c@u@c)/((c@pos)*(c@neg));assert abs(actual-expect)<1e-12
tests['paired_region_weighted_auc_identity_20_draws']=True
# Actual membership manifests; source IDs are auditable, not only asserted labels.
manifest=P/'membership';manifest.mkdir(exist_ok=True)
audit=[]
for label,folder,buffer_km,geo_complete in [('primary','region_holdout',0,False),('buffer50','region_buffer50',50,True),('buffer_reference','region_buffer50_reference',0,True)]:
    prediction=pd.read_parquet(P/folder/'spatial_predictions_frame.parquet')
    assert len(prediction)==66395 and prediction.PWSID.nunique()==66395
    for held in range(1,11):
        test=F[F.block==held];train=source[source.block!=held].copy();before=len(train)
        if geo_complete:
            train=train[train[['lat','lon']].notna().all(axis=1)]
        if buffer_km:
            train=train.loc[~st.source_buffer_exclusion(train,test,buffer_km)]
        r=json.loads((P/folder/f'fold_{held:02d}_audit.json').read_text())
        assert len(train)==r['train_n'] and not train.PWSID.isin(test.PWSID).any()
        assert hashlib.sha256('\n'.join(sorted(train.PWSID)).encode()).hexdigest()==r['source_ids_sha256']
        train[['PWSID','block']].to_csv(manifest/f'{label}_region_{held:02d}_source_ids.csv',index=False)
        min_distance=np.nan
        if buffer_km:
            target=test[test.y_mcl.notna()&test.lat.notna()&test.lon.notna()]
            tree=BallTree(np.radians(target[['lat','lon']]),metric='haversine')
            distances,_=tree.query(np.radians(train[['lat','lon']]),k=1);min_distance=float(distances.min()*6371.)
            assert min_distance>50-1e-8
        for trblock,group in train.groupby('block'):
            reference=train[train.block!=trblock]
            assert held not in reference.block.unique() and trblock not in reference.block.unique()
            audit.append(dict(regime=label,held_region=held,training_region=int(trblock),source_n=len(group),training_neighbour_reference_n=len(reference),held_region_in_reference=0,own_region_in_reference=0,min_buffer_distance_km=min_distance))
tests['actual_30_fold_memberships_and_50km_distance_checks']=True
pd.DataFrame(audit).to_csv(P/'independence_reference_counts.csv',index=False)
for f in ['region_holdout','region_buffer50','region_coordinate_diagnostic','region_buffer50_reference']:
    d=json.loads((P/f/'analysis_completion.json').read_text())
    assert d['unchanged_input_sha256'] and d['all_fold_audits_passed']
    assert hashlib.sha256((paths.DERIVED/'frame_systems.parquet').read_bytes()).hexdigest()==d['frame_sha256']
tests['frozen_input_sha256_unchanged']=True
spec={'tests':tests,'all_passed':all(tests.values()),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lightgbm.__version__,'sklearn':sklearn.__version__},
      'scope':'Scientific information-boundary and weighted-performance identities, not proof of population representativeness or external field performance.'}
(P/'spatial_validation_checks.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print(json.dumps(spec,indent=2))
