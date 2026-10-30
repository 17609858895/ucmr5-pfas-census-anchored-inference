"""Restrict whole reported-event sets to spans <=366 days; frozen f unchanged."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from project_paths import D, R
from s25_weight_sensitivity import estimate, weights_for

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events',type=Path,default=D/'loc_event.parquet')
    ap.add_argument('--frame',type=Path,default=D/'small_frame_anchor.parquet')
    ap.add_argument('--inventory',type=Path,default=D/'frame_systems.parquet')
    ap.add_argument('--output',type=Path,default=R/'review_sensitivity')
    args=ap.parse_args();P=args.output;P.mkdir(parents=True,exist_ok=True)
    le=pd.read_parquet(args.events)
    sm=pd.read_parquet(args.frame)
    frame=pd.read_parquet(args.inventory)
    key=['PWSID','FacilityID','SamplePointID']
    le['sw']=le.WaterType.isin(['SW','GU','MX']);n=le.groupby(key).SampleEventCode.transform('nunique')
    le=le[n.eq(np.where(le.sw,4,2))].copy();le['pf']=le.v_PFOA.fillna(0);le['ps']=le.v_PFOS.fillna(0)
    loc=le.groupby(key).agg(n_events=('SampleEventCode','nunique'),start=('date','min'),end=('date','max'),pfoa=('pf','mean'),pfos=('ps','mean')).reset_index()
    loc['y']=((loc.pfoa>=4.05)|(loc.pfos>=4.05)).astype(int);loc['span_days']=(loc.end-loc.start).dt.days
    old_y=loc.groupby('PWSID').y.max();ret=loc[loc.span_days<=366].copy();new_y=ret.groupby('PWSID').y.max()
    smallset=set(sm.PWSID);currentset=set(frame.PWSID)
    baseline=sm.PWSID.map(old_y);assert np.allclose(baseline.fillna(-1),sm.y_mcl.fillna(-1))
    compare=[]
    for group,ids in [('all_reported_systems',set(old_y.index)),('matched_active_inventory',currentset),('current_small_inventory',smallset)]:
        before=loc[loc.PWSID.isin(ids)];after=ret[ret.PWSID.isin(ids)]
        b=old_y.reindex(sorted(ids)).dropna();a=new_y.reindex(sorted(ids)).dropna()
        common=b.index.intersection(a.index)
        compare.append({'group':group,'locations_before':len(before),'locations_after':len(after),'systems_before':len(b),'systems_after':len(a),'positive_systems_before':int(b.sum()),'positive_systems_after':int(a.sum()),'positive_to_negative_among_retained':int(((b.loc[common]==1)&(a.loc[common]==0)).sum()),'positive_systems_lost_all_locations':int(b.loc[b.index.difference(a.index)].sum()),'dropped_locations':len(before)-len(after)})
    sm2=sm.copy();sm2['y_mcl']=sm2.PWSID.map(new_y)
    estimates=[]
    for name,d in [('primary_reported_event_counts',sm),('span_le366_days',sm2)]:
        w=weights_for(d,'cell');z=estimate(d,w)
        estimates.append({'scheme':name,'n_observed':int(d.y_mcl.notna().sum()),'N':len(d),**z})
    pd.DataFrame(compare).to_csv(P/'timing_eligibility_population_changes.csv',index=False)
    pd.DataFrame(estimates).to_csv(P/'timing_burden_sensitivity.csv',index=False)
    loc[loc.span_days>366].to_csv(P/'locations_span_over366days.csv',index=False)
    info={'definition':'Exclude entire retained location-event set when last minus first event date exceeds 366 days; do not enforce quarter or half-year spacing; not a compliance adjudication.','primary_event_sets_unchanged':True,'LightGBM_frozen_predictions_unchanged':True,'comparison':compare,'burden':estimates}
    (P/'timing_sensitivity_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
