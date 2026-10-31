from strengthening import paths
"""Freeze canonical source-verified v2 protocol, source hashes, and cohort QA.

No private contact columns from the official general/source inventories are
retained in the derived crosswalk. Raw public source workbooks stay local.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd
from . import build_external_cohorts as c
P=paths.EXTERNAL_RESULTS; O=paths.EXTERNAL_RESULTS; RAW=paths.WA_RAW

sources=[
 ('PFAS_Downloadable_Data.csv','https://doh.wa.gov/sites/default/files/Data/Auto-Uploads/PFAS/PFAS_Downloadable_Data.csv'),
 ('PFAS_Data_Dictionary.csv','https://doh.wa.gov/sites/default/files/Data/Auto-Uploads/PFAS/PFAS_Data_Dictionary.csv'),
 ('wa_general.xlsx','https://doh.wa.gov/sites/default/files/2025-08/ageneral.xlsx'),
 ('wa_sources.xlsx','https://doh.wa.gov/sites/default/files/2023-01/combinedsources.xlsx')]
manifest=[]
for name,url in sources:
 f=RAW/name;manifest.append({'file':name,'url':url,'retrieved_utc_date':'2026-10-31','bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(RAW/'source_manifest_complete.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

names=pd.read_csv(O/'wa_exact_id_crosswalk.csv',dtype=str)
general=pd.read_excel(RAW/'wa_general.xlsx',dtype=str)
general=general.rename(columns={'WS ID':'WSID','WS Name':'WSName'})
general['WSID']=general.WSID.str.zfill(5)
cross=names.merge(general[['WSID','WSName']],left_on='local_id',right_on='WSID',how='left',validate='many_to_one')
def normal(x):return re.sub('[^A-Z0-9]','',str(x).upper())
cross['normalized_name_exact']=cross['Water System Name'].map(normal).eq(cross.WSName.map(normal))
cross.to_csv(O/'wa_official_general_id_crosswalk_QA.csv',index=False)
idqa={'source_unique_systems':len(names),'official_general_ID_matches':int(cross.WSID.notna().sum()),'official_general_unmatched_IDs':cross.loc[cross.WSID.isna(),'PWSID'].tolist(),'exact_normalized_name_matches':int(cross.normalized_name_exact.sum()),'matching_rule':'Exact local five-character WSID embedded in WA53+WSID federal PWSID; official source number exact join. Names used only for QA, never fuzzy matching.','dictionary_fields':'Water System ID = ID for the drinking water source; Water Source = Number of the drinking water source. Official GroupA general/source inventories provide WSID and SrcNum.','unmatched_handling':'Require exact match to current small-system federal inventory and, for WA, verified Ground Water*/Surface source type; unresolved source type is excluded.'}
(O/'wa_id_crosswalk_QA.json').write_text(json.dumps(idqa,indent=2),encoding='utf-8')

S=pd.read_csv(O/'external_schedule_systems.csv');flows=[]
for state,loader in [('MI',c.mi_events),('WA',c.wa_events)]:
 d=loader();m=d.merge(c.sm[['PWSID','sw']],on='PWSID',how='inner')
 stages=[('paired distinct dates, all available',d),('current small-system inventory exact-ID matched',m),('2023-01-01 to 2026-06-30',m[m.date.between(c.START,c.END)])]
 q=stages[-1][1];q=q[~q.PWSID.isin(c.excluded)];stages.append(('all UCMR5 PWSIDs excluded',q))
 if state=='WA':
  src=pd.read_csv(O/'wa_location_source_crosswalk.csv',dtype={'location':str})[['PWSID','location','need_actual']]
  q=q.merge(src,on=['PWSID','location'],validate='many_to_one');q=q[q.need_actual.gt(0)]
  stages.append(('verified eligible per-source type',q))
 for stage,h in stages:
  flows.append({'state':state,'stage':stage,'paired_dates':len(h),'systems':h.PWSID.nunique(),'locations':len(h[['PWSID','location']].drop_duplicates())})
 loc=pd.read_csv(O/(state+('_source_verified_locations.csv' if state=='WA' else '_strict_locations.csv')))
 sys=S[S.external_state.eq(state)]
 flows.append({'state':state,'stage':'first 2/4 dates within 366 days','paired_dates':int(loc.k.sum()),'systems':len(sys),'locations':len(loc),'positive_systems':int(sys.y_full.sum()),'full_event_cost':int(sys.c_full.sum())})
pd.DataFrame(flows).to_csv(O/'external_cohort_flow_counts.csv',index=False)

protocol=c.log.copy()
protocol.update({'version':'v2 source-verified, frozen 2026-10-31','source_class_limitation':'WA: verified official GroupA per-source classification, Ground Water* => two events, Surface => four; Intertie, SeaWater, RIGallery, Other or unmatched types excluded. MI: per-location source class unavailable, current SDWIS system source class is a proxy. No claim of annual regulatory-compliance schedules.','event_selection':'At each location use first k distinct paired collection dates in 2023-01-01–2026-06-30; k=2 GW or 4 surface; first-k span <=366 days; same-day samples averaged before date selection; no outcome-based date selection. WA actual per-source k, MI system source proxy.','endpoint':'Each analyte concentration below 4 ng/L set to zero; arithmetic mean over fixed selected dates; PWSID positive if either analyte mean >=4.05 ng/L at any eligible location. WA uses published integer concentrations, so this is a published-value endpoint with limited precision.','WA_duplicate_identity':'Use explicitly state-action-level-coded PFOA/PFOS records. Same-name PFOS records without SAL metadata have no resolved chemical/isomer identity and are not silently averaged. Max/sum alternatives are sensitivity analyses. Public semantic schema distinguishes analytes with versus without state action levels.','cohort_selection_boundary':'Repeat-testing eligibility is associated with initial concentrations. Cohort fractions are conditional on observed screening/follow-up and current inventory matching; neither state burden nor probability-sampling estimates. Initial-screen and calendar/spacing checks diagnose selection and timing.','prediction_independence':'All 10,313 PWSIDs with any UCMR5 record excluded from external cohorts. Original operational predictions use frozen larger-system models and nearby larger-system observations; region-excluded alternatives exclude all larger-system labels and neighbour labels from target EPA region. No external outcomes used in fit, calibration or model selection.','uncertainty':'AUC/Brier and prediction-error intervals use 2000 PWS bootstrap or 0.5-degree administrative-centroid block bootstrap draws; primary AUC interval is spatial-block. These describe selected-cohort uncertainty, not national/state probability-sampling confidence. Ratio whiskers propagate only original national calibration-factor uncertainty.','quantization_boundary':'Dashboard/public schema has integer-valued public PFAS Level Sort and string PFAS Level. No unrounded concentration, assay-specific reporting limit or documented nearest-integer rounding rule located. Hypothetical +/-0.5 nearest-integer ranges and exact rank-based AUC extrema are assumption scenarios, not identified true-label bounds or confidence intervals.','calendar_sensitivity':'Earliest complete 2023–2025 calendar year, first paired observation in both GW half-years or each of four surface quarters; no outcome selection; comparison on exact shared location universe distinguishes definition change from selected-cohort composition.','spacing_sensitivity':'Restrict original first-k event sets to span >=150 GW / >=240 surface days, still <=366; do not re-select dates or claim regulatory annual qualification.'})
for state,g in S.groupby('external_state'):
 protocol['states'][state].update({'strict_n':len(g),'strict_positive':int(g.y_full.sum()),'R12_missed':int((g.y_full-g.y_R12).sum()),'c_full_sum':int(g.c_full.sum()),'groundwater_n':int(g.sw.eq(0).sum()),'groundwater_positive':int(g.loc[g.sw.eq(0),'y_full'].sum())})
protocol['WA_source_inventory_QA']=json.loads((O/'wa_source_verification_impact.json').read_text(encoding='utf-8'))
screen=pd.read_csv(O/'initial_screening_systems.csv',parse_dates=['first_date','followup_observation_end'])
protocol['followup_opportunity']={st:{'effective_observation_end':str(g.followup_observation_end.iloc[0].date()),'latest_early_index_date':str((g.followup_observation_end.iloc[0]-pd.Timedelta(days=366)).date()),'early_systems':int(g.early_index_366day_opportunity.sum())} for st,g in screen.groupby('external_state')}
(O/'cohort_protocol_and_counts.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')

# Verify immutable cohort independence and arithmetic across all final cohorts.
checks=[]
for filename in ['external_schedule_systems.csv','calendar_aligned_systems.csv','MI_spacing_sensitivity_systems.csv','WA_spacing_sensitivity_systems.csv']:
 q=pd.read_csv(O/filename)
 assert not set(q.PWSID)&c.excluded
 assert not q.PWSID.duplicated().any()
 assert (q.y_R12<=q.y_full).all() and (q.c_R12<=q.c_full).all()
 assert q[['f','u','P']].notna().all().all()
 checks.append({'file':filename,'rows':len(q),'positive':int(q.y_full.sum()),'sha256':hashlib.sha256((O/filename).read_bytes()).hexdigest(),'all_ucmr_exclusion':'PASS','unique_PWSID':'PASS','R12_monotonicity':'PASS','predictor_complete':'PASS'})
(O/'canonical_cohort_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(idqa,indent=2));print(pd.DataFrame(flows).to_string(index=False));print('Canonical cohort checks PASS')
