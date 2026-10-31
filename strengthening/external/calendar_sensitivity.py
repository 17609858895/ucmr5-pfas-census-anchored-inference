from .protocol_core import first_complete_calendar
from strengthening import paths
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from . import build_external_cohorts as c
P=paths.EXTERNAL_RESULTS;O=paths.EXTERNAL_RESULTS;systems=[];flows=[];paired=[];transitions=[]
original=pd.read_csv(O/'external_schedule_systems.csv')
for state,loader in [('MI',c.mi_events),('WA',c.wa_events)]:
    d=loader().merge(c.sm[['PWSID','sw']],on='PWSID',how='inner');d=d[~d.PWSID.isin(c.excluded)&d.date.between('2023-01-01','2025-12-31')].copy()
    d['need_actual']=np.where(d.sw.eq(1),4,2)
    if state=='WA':
        source=pd.read_csv(O/'wa_location_source_crosswalk.csv',dtype={'location':str})[['PWSID','location','need_actual']]
        d=d.drop(columns='need_actual').merge(source,on=['PWSID','location'],validate='many_to_one');d=d[d.need_actual.gt(0)]
    d=d.drop(columns='sw');selected=[]
    for (pid,loc),g in d.groupby(['PWSID','location'],sort=True):
        q=first_complete_calendar(g)
        if q is not None:selected.append(q)
    selected=pd.concat(selected,ignore_index=True);selected.to_csv(O/(state+'_calendar_selected_events_input.csv'),index=False)
    ss=c.build_schedules(selected,state,tag='calendar_aligned');ss['source_class_basis']='officialWA per-source inventory' if state=='WA' else 'SDWIS systemsourceproxy';systems.append(ss)
    cl=pd.read_csv(O/(state+'_calendar_aligned_locations.csv'));count_loc=pd.read_csv(O/(state+('_source_verified_locations.csv' if state=='WA' else '_strict_locations.csv')))
    both=count_loc.merge(cl,on=['PWSID','location'],suffixes=('_count','_calendar'),validate='one_to_one')
    # Comparing the same location universe avoids conflating observation-rule
    # changes with a change in the set of eligible locations or systems.
    sy=both.groupby('PWSID').agg(y_count=('y_full_count','max'),y_calendar=('y_full_calendar','max'),n_shared_locations=('location','size')).reset_index().merge(c.sm,on='PWSID')
    sy['external_state']=state;paired.append(sy)
    n01=int(((sy.y_count==0)&(sy.y_calendar==1)).sum());n10=int(((sy.y_count==1)&(sy.y_calendar==0)).sum());rng=np.random.default_rng(340+len(sy));delta=sy.y_calendar.to_numpy()-sy.y_count.to_numpy();idx=rng.integers(0,len(sy),(4000,len(sy)));lo,hi=np.percentile(delta[idx].mean(axis=1),[2.5,97.5])
    transitions.append({'state':state,'shared_systems':len(sy),'shared_locations':len(both),'count_positive_on_shared_locations':int(sy.y_count.sum()),'calendar_positive_on_shared_locations':int(sy.y_calendar.sum()),'became_positive':n01,'became_negative':n10,'paired_prevalence_change':delta.mean(),'paired_change_lo':lo,'paired_change_hi':hi,'mcnemar_exact_p':binomtest(n01,n01+n10,.5).pvalue if n01+n10 else 1})
    for name,h in [('count_window',original[original.external_state.eq(state)]),('calendar_aligned',ss)]:
        flows.append({'state':state,'cohort':name,'n':len(h),'positives':int(h.y_full.sum()),'prevalence':h.y_full.mean(),'mean_f':h.f.mean(),'auc':roc_auc_score(h.y_full,h.f),'full_cost':int(h.c_full.sum()),'R1_missed':int((h.y_full-h.y_R1).sum()),'R12_missed':int((h.y_full-h.y_R12).sum()),'R12_cost':int(h.c_R12.sum())})
pd.concat(systems,ignore_index=True).to_csv(O/'calendar_aligned_systems.csv',index=False)
pd.concat(paired,ignore_index=True).to_csv(O/'calendar_count_paired_shared_location_systems.csv',index=False)
pd.DataFrame(flows).to_csv(O/'calendar_cohort_comparison.csv',index=False);pd.DataFrame(transitions).to_csv(O/'calendar_count_paired_transitions.csv',index=False)
(O/'calendar_protocol.json').write_text(json.dumps({'years':[2023,2024,2025],'rule':'At each location choose earliestcompletecalendaryear withpairedsamplesinbothhalves(GW)orallfourquarters(Surface);takefirstpairedsampleineachperiod.Nooutcomeselection.Noregulation-complianceclaim.','paired_analysis':'Compare count-window and calendar endpoint on the exactsamePWSID/locationintersection,thenaggregateany-locationatPWSID;4000pairedPWSbootstrapandexactMcNemar.','interpretation':'All-cohort changes combine selection and observation-rule differences; pairedshared-location comparison holds location/systemuniverse fixed.'},indent=2),encoding='utf-8')
print(pd.DataFrame(flows).round(4).to_string(index=False));print(pd.DataFrame(transitions).round(5).to_string(index=False))
