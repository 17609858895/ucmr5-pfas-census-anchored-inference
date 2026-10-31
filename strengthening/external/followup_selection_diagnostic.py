from strengthening import paths
from .protocol_core import effective_observation_end
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.stats.contingency import odds_ratio
from . import build_external_cohorts as c
P=paths.EXTERNAL_RESULTS;O=paths.EXTERNAL_RESULTS;END=pd.Timestamp(c.END);ALL=[];SYS=[];SUMMARY=[]
for state,loader in [('MI',c.mi_events),('WA',c.wa_events)]:
 raw=loader();state_end=effective_observation_end(raw.date,END)
 d=raw.merge(c.sm[['PWSID','sw','P','pop','f','u','src','cell','lat','lon','PRIMACY_AGENCY_CODE']],on='PWSID',how='inner')
 d=d[(d.date>=c.START)&(d.date<=c.END)&~d.PWSID.isin(c.excluded)].copy();d['need']=np.where(d.sw==1,4,2)
 if state=='WA':
  source=pd.read_csv(O/'wa_location_source_crosswalk.csv',dtype={'location':str})[['PWSID','location','need_actual']]
  d=d.merge(source,on=['PWSID','location'],validate='many_to_one');d=d[d.need_actual.gt(0)];d['need']=d.need_actual.astype(int)
 for a in ['PFOA','PFOS']:d[a]=d[a].where(d[a]>=4,0)
 loc=[]
 for (pid,name),g in d.groupby(['PWSID','location'],sort=True):
  g=g.sort_values('date');q=g.head(int(g.need.iloc[0]));v=q[['PFOA','PFOS']].to_numpy(float)
  ok=int(len(q)==int(g.need.iloc[0]) and (q.date.max()-q.date.min()).days<=366)
  first=g.iloc[0]
  loc.append({'external_state':state,'PWSID':pid,'location':name,'first_date':first.date,'first_detect4':int(max(first.PFOA,first.PFOS)>=4),'first_exceed405':int(max(first.PFOA,first.PFOS)>=4.05),'eligible':ok,'paired_dates_in_window':len(g),'need':int(g.need.iloc[0]),'followup_observation_end':state_end,'early_index_366day_opportunity':int(first.date<=state_end-pd.Timedelta(days=366)),'P':first.P,'sw':first.sw})
 loc=pd.DataFrame(loc);ALL.append(loc)
 # Initial screening is explicitly a single-event endpoint: sources collected
 # on the earliest system date are combined by maximum, never by outcomes.
 firstdate=d.groupby('PWSID').date.transform('min');idx=d[d.date.eq(firstdate)].copy()
 idx['y_first_screen']=(idx[['PFOA','PFOS']].max(axis=1)>=4.05).astype(int)
 idx['det_first_screen']=(idx[['PFOA','PFOS']].max(axis=1)>=4).astype(int)
 sy=idx.groupby('PWSID').agg(first_date=('date','min'),y_first_screen=('y_first_screen','max'),det_first_screen=('det_first_screen','max')).reset_index()
 eligibility=loc.groupby('PWSID').eligible.max();sy['strict_eligible']=sy.PWSID.map(eligibility)
 sy=sy.merge(c.sm,on='PWSID');sy['external_state']=state;sy['followup_observation_end']=state_end;sy['early_index_366day_opportunity']=(sy.first_date<=state_end-pd.Timedelta(days=366)).astype(int);SYS.append(sy)
 for unit,g in [('location',loc),('system',sy.rename(columns={'strict_eligible':'eligible','det_first_screen':'first_detect4','y_first_screen':'first_exceed405'}))]:
  if unit=='system':g['early_index_366day_opportunity']=(g.first_date<=state_end-pd.Timedelta(days=366)).astype(int)
  for subgroup,h in [('all',g),('early_index',g[g.early_index_366day_opportunity.eq(1)])]:
   for exposure in ['first_detect4','first_exceed405']:
    tab=pd.crosstab(h[exposure],h.eligible).reindex(index=[0,1],columns=[0,1],fill_value=0)
    # OR of eligibility for positive vs negative initial observations.
    table=np.array([[tab.loc[1,1],tab.loc[1,0]],[tab.loc[0,1],tab.loc[0,0]]])
    est=odds_ratio(table,kind='conditional');iv=est.confidence_interval()
    SUMMARY.append({'state':state,'unit':unit,'subgroup':subgroup,'initial_condition':exposure,'n':len(h),'negative_n':int(tab.loc[0].sum()),'negative_eligible':int(tab.loc[0,1]),'negative_eligible_fraction':tab.loc[0,1]/max(tab.loc[0].sum(),1),'positive_n':int(tab.loc[1].sum()),'positive_eligible':int(tab.loc[1,1]),'positive_eligible_fraction':tab.loc[1,1]/max(tab.loc[1].sum(),1),'eligibility_odds_ratio':est.statistic,'odds_ratio_ci_lo':iv.low,'odds_ratio_ci_hi':iv.high})
pd.concat(ALL,ignore_index=True).to_csv(O/'followup_eligibility_by_location.csv',index=False)
pd.concat(SYS,ignore_index=True).to_csv(O/'initial_screening_systems.csv',index=False)
summary=pd.DataFrame(SUMMARY);summary.to_csv(O/'followup_eligibility_selection_summary.csv',index=False)
flow=pd.concat(SYS).groupby('external_state').agg(initial_systems=('PWSID','size'),initial_positive=('y_first_screen','sum'),initial_detect4=('det_first_screen','sum'),strict_eligible=('strict_eligible','sum'))
flow.to_csv(O/'initial_vs_repeat_sampled_counts.csv');print(flow.to_string());print(summary.query("unit=='system' and subgroup=='early_index' and initial_condition=='first_detect4'").round(3).to_string(index=False))
