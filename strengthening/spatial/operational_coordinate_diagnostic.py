from strengthening import paths
"""Matched available-national-layer counterpart to the geographic-ID ablation.

Only larger UCMR5 outcomes supervise the models; no external state outcomes
or small-system outcomes are fitted. These are post-hoc developmental models,
fixed before their new-state comparison, not a replacement chosen by test AUC.
"""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd,lightgbm as lgb
from . import spatial_transfer as st
def run(frame_path,out):
    out.mkdir(parents=True,exist_ok=True);f=st.setup(pd.read_parquet(frame_path));train=f[f.size_cat.isin(['medium','large'])&f.y_mcl.notna()].copy()
    tr=train.join(st.nb_features(train,train));target=f.join(st.nb_features(f,train));tr,target,cats=st.category_maps(tr,target)
    p=target[['PWSID','block','PRIMACY_AGENCY_CODE','STATE_CODE','size_cat','pop','lat','lon','y_mcl','y_det']].copy()
    for name,cols in {**st.SETS,'coordinates':st.ATTR+['lat','lon'],'coordinates_neighbours':st.ATTR+['lat','lon']+st.NB}.items():
        for seed in [0,1,2]:
            model=lgb.LGBMClassifier(**st.GB,random_state=seed).fit(tr[cols],tr.y_mcl.astype(int))
            p[f'{name}_seed{seed}']=model.predict_proba(target[cols])[:,1]
        p[name]=p[[f'{name}_seed{s}' for s in [0,1,2]]].mean(axis=1)
    p.to_parquet(out/'operational_coordinate_predictions.parquet',index=False);p.to_csv(out/'operational_coordinate_predictions.csv',index=False)
    (out/'operational_diagnostic_specification.json').write_text(json.dumps({'source_frame_sha256':hashlib.sha256(frame_path.read_bytes()).hexdigest(),'source_n':len(train),'source_only':'All9229observedlarger UCMR5 systems; no small/external outcomes fitted.','neighbour_reference':'Allmeasuredlargersystems; ownPWSIDexcluded. Availablelayerregime.','feature_change':'OmitstateandEPAregion IDs; keepcoordinatesandsameattributes.','seeds':[0,1,2],'status':'Posthocdevelopmentaldiagnosticfollowingcoldadministrativecategoryfailure; no externaltestselection.','warning':'Larger-system predictions in this file are fitted, not OOS; do not report their training AUC as validation.'},indent=2),encoding='utf-8')
    print('Operational diagnostic fullframe predictions',len(p),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--frame',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();run(x.frame,x.out)
