from .protocol_core import source_event_count
from strengthening import paths
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from . import build_external_cohorts as c
P=paths.EXTERNAL_RESULTS;O=paths.EXTERNAL_RESULTS
old=O/'system_source_proxy_v1';old.mkdir(exist_ok=True)
for name in ['external_schedule_systems.csv','WA_strict_events.csv','WA_strict_locations.csv','external_schedule_predictions.csv','cohort_protocol_and_counts.json','strict_external_metrics_bootstrap.csv','followup_eligibility_selection_summary.csv']:
    if (O/name).exists() and not (old/name).exists():shutil.copy2(O/name,old/name)
src=pd.read_excel(paths.WA_RAW/'wa_sources.xlsx',dtype=str)
src=src[src['WS Grp'].eq('A')].copy();src['PWSID']='WA53'+src['WS ID'].str.zfill(5);src['location']=src.SrcNum.str.zfill(2)
assert src.groupby(['PWSID','location']).SrcType.nunique().max()==1
src=src.drop_duplicates(['PWSID','location'])
src['need_actual']=src.SrcType.map(source_event_count)
events=c.wa_events();e=events.merge(src[['PWSID','location','SrcType','need_actual']],on=['PWSID','location'],how='left',validate='many_to_one')
e=e.merge(c.sm[['PWSID','sw']],on='PWSID',how='inner');e=e[~e.PWSID.isin(c.excluded)&e.date.between(c.START,c.END)]
e['need_proxy']=np.where(e.sw.eq(1),4,2)
q=e[['PWSID','location','SrcType','need_actual','need_proxy']].drop_duplicates()
q.to_csv(O/'wa_location_source_crosswalk.csv',index=False)
summary={'official_groupA_source_rows':len(src),'independent_candidate_locations':len(q),'verified_eligible_type_locations':int(q.need_actual.gt(0).sum()),'unmatched_locations':int(q.need_actual.isna().sum()),'unclassified_known_type_locations':int(q.need_actual.eq(0).sum()),'proxy_vs_verified_disagreements':int((q.need_actual.gt(0)&q.need_actual.ne(q.need_proxy)).sum()),'unknown_type_counts':q[~q.need_actual.gt(0)].SrcType.fillna('not matched').value_counts().to_dict()}
e=e[e.need_actual.gt(0)].drop(columns=['sw'])
wa=c.build_schedules(e,'WA',tag='source_verified');oldsys=pd.read_csv(old/'external_schedule_systems.csv');mi=oldsys[oldsys.external_state.eq('MI')].copy()
mi['source_class_basis']='current SDWIS system source';wa['source_class_basis']='official WA GroupA per-source inventory'
s=pd.concat([mi,wa],ignore_index=True);s.to_csv(O/'external_schedule_systems.csv',index=False)
loc=pd.read_csv(O/'WA_source_verified_locations.csv');ev=pd.read_csv(O/'WA_source_verified_events.csv');ev['date']=pd.to_datetime(ev.date)
minspan=loc.k.map({2:150,4:240});loc['spacing_sensitivity_eligible']=loc.span_days.ge(minspan)
sp=loc[loc.spacing_sensitivity_eligible]
cols=['y_full','y_threshold4','y_R1','y_R2','y_R12','y_quantization_lo','y_quantization_hi','y_PFOS_allmax','y_PFOS_allsum']
agg={col:'max' for col in cols};agg.update({col:'sum' for col in ['c_full','c_R1','c_R2','c_R12']})
spacing=sp.groupby('PWSID',as_index=False).agg(agg).merge(c.sm,on='PWSID');spacing['external_state']='WA';spacing.to_csv(O/'WA_spacing_sensitivity_systems.csv',index=False)
summary.update({'proxy_n':int(oldsys.external_state.eq('WA').sum()),'proxy_positive':int(oldsys[oldsys.external_state.eq('WA')].y_full.sum()),'verified_n':len(wa),'verified_positive':int(wa.y_full.sum()),'verified_full_cost':int(wa.c_full.sum()),'verified_R12_missed':int((wa.y_full-wa.y_R12).sum()),'same_IDs_as_proxy':len(set(wa.PWSID)&set(oldsys[oldsys.external_state.eq('WA')].PWSID)),'spacing_sensitivity_n':len(spacing),'spacing_sensitivity_positive':int(spacing.y_full.sum())})
(O/'wa_source_verification_impact.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
