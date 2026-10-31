from strengthening import paths
"""Independent small-cohort predictions for ratio calibration outside target region.

Reuses exactly the fixed outer-region model. No held-region larger outcome or
small outcome is fitted. Both neighbour-availability variants are supplied so
the source/cold information difference is explicit, not selected by test results.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,lightgbm as lgb
from . import spatial_transfer as st

def run(frame_path,out):
    out.mkdir(parents=True,exist_ok=True);f=st.setup(pd.read_parquet(frame_path));source=f[f.size_cat.isin(['medium','large'])&f.y_mcl.notna()].copy()
    labels=f[(f.size_cat=='small')&f.y_mcl.notna()].copy();assert len(labels)==970
    rows=[]
    for held in [1,5,10]:
        train=source[source.block!=held].copy();train=train.join(st.training_nb(train))
        cal=labels[labels.block!=held].copy()
        for mode in ['available_neighbours','cold_own_region']:
            feat=st.nb_features(cal,train) if mode=='available_neighbours' else None
            if feat is None:
                feat=pd.DataFrame(np.nan,index=cal.index,columns=st.NB)
                for block,g in cal.groupby('block'):feat.loc[g.index]=st.nb_features(g,train[train.block!=block])
            test=cal.join(feat);tr,te,_=st.category_maps(train,test)
            pred=te[['PWSID','block','y_mcl','pop','GW_SW_CODE']].copy();pred['held_region']=held;pred['neighbour_regime']=mode
            for name,cols in st.SETS.items():
                # Attribute/geography fits are identical between the two modes;
                # keeping both rows makes their equality directly testable.
                for seed in [0,1,2]:
                    model=lgb.LGBMClassifier(**st.GB,random_state=seed).fit(tr[cols],tr.y_mcl.astype(int))
                    pred[f'{name}_seed{seed}']=model.predict_proba(te[cols])[:,1]
                pred[name]=pred[[f'{name}_seed{s}' for s in [0,1,2]]].mean(axis=1)
            rows.append(pred);print('Calibration predictions',held,mode,len(pred),flush=True)
    result=pd.concat(rows,ignore_index=True);result.to_parquet(out/'calibration_by_held_region.parquet',index=False);result.to_csv(out/'calibration_by_held_region.csv',index=False)
    for held in [1,5,10]:
        for name in ['attributes','geography']:
            x=result[result.held_region==held].pivot(index='PWSID',columns='neighbour_regime',values=name)
            assert np.max(np.abs(x.iloc[:,0]-x.iloc[:,1]))<1e-12
    (out/'calibration_independence.json').write_text(json.dumps({'held_regions':[1,5,10],'large_training':'outsideheldregiononly','calibration_labels':'970observedsmallsystems outsideheldregiononly; none fitted in larger-modeltraining.','available_neighbours':'Calibrationrows use all largerlabels outsideheldregion, may include their own region.','cold_own_region':'Calibrationneighbours additionally omit ownregion, approximating coldlocal-labelavailability.','weights':'Calibrationusers must compute8cell expansion weights from outsideheldregion inventory/observed counts, not use heldregionoutcomes.','selection':'Alternativecalibrationregimes are supplied before externalvalidation results; donot choose best using externaloutcomes.'},indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frame',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.frame,a.out)
