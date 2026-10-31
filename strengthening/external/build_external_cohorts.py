from .protocol_core import washington_local_id, exclude_known_ids, first_count_window, retained_mean_outcome
from strengthening import paths
"""Prespecified independent state-cohort and fixed first-schedule construction."""
from pathlib import Path
import json,hashlib,re
import numpy as np,pandas as pd
P=paths.EXTERNAL_RESULTS;ROOT=paths.WORKDIR
BASE=paths.WORKDIR;DATA=paths.DERIVED;EXT=paths.EXTERNAL_RAW
OUT=paths.EXTERNAL_RESULTS;OUT.mkdir(exist_ok=True)
START='2023-01-01';END='2026-06-30';MAXSPAN=366
sm=pd.read_parquet(DATA/'small_frame_anchor.parquet')
ucmr=pd.read_parquet(DATA/'ucmr5_all.parquet',columns=['PWSID','PWSName']);excluded=set(ucmr.PWSID)
log={'window':[START,END],'max_span_days':MAXSPAN,'event_selection':'First k distinct paired collection dates at each location in the window; k=2 for the current system groundwater class,4 for surface class. No outcome-based selection. Multiple same-day samples are averaged before selecting dates. Only first-k span <=366 retained.','source_class_limitation':'State downloads lack per-location GW/SW class; k is assigned from matched SDWIS system source class. Groundwater-only results are separately provided.','all_ucmr5_pwsids_excluded':len(excluded),'states':{}}

def parse_num(s):
    text=s.fillna('').astype(str).str.strip()
    nd=text.str.startswith('<')|text.str.upper().isin(['ND','NONE DETECTED','NON-DETECT'])
    return pd.to_numeric(text.where(~nd,'0'),errors='coerce')

def mi_events():
    d=pd.read_csv(EXT/'michigan_egle_20261025/mi_pws_pfas_results.csv',dtype=str,encoding='utf-8-sig')
    d['PWSID']='MI'+d.WSSN.str.zfill(7);d['location']=d.LocName;d['date']=pd.to_datetime(d.SampleDate,utc=True).dt.tz_localize(None)
    for c in ['PFOA','PFOS']:d[c]=parse_num(d[c])
    before=len(d);d=d.dropna(subset=['PFOA','PFOS','date','location']).drop_duplicates(['PWSID','location','date','SysSampleCode','PFOA','PFOS'])
    log['states']['MI']={'raw_rows':before,'paired_nonmissing_rows':len(d),'sample_date_min':str(d.date.min()),'sample_date_max':str(d.date.max()),'native_reporting_limit':'Explicit<2 values; numericvalues below4 are recensored to0.'}
    return d.groupby(['PWSID','location','date'],as_index=False).agg(PFOA=('PFOA','mean'),PFOS=('PFOS','mean'))

def wa_events():
    d=pd.read_csv(paths.WA_RAW/'PFAS_Downloadable_Data.csv',dtype=str)
    d=d[d['PFAS Measure'].str.contains(r'\(PFOA\)|\(PFOS\)',regex=True,na=False)].copy()
    d['local_id']=d['Water System ID'].map(washington_local_id)
    d['PWSID']='WA53'+d.local_id;d['location']=d['Water Source'].str.replace('="','',regex=False).str.replace('"','',regex=False)
    d['date']=pd.to_datetime(d['Testing Date']);d['analyte']=d['PFAS Measure'].str.extract(r'\((PFOA|PFOS)\)')[0]
    d['value']=pd.to_numeric(d['PFAS Level'].str.replace(' ng/L','',regex=False),errors='coerce')
    d.loc[d['PFAS Level'].eq('None Detected'),'value']=0.
    original=d.copy()
    # Retain the explicitly action-level-coded regulated analyte records.
    # Public export also contains same-name PFOS records without this metadata;
    # their identity is undocumented, so they cannot be silently averaged.
    d=d[d['State Action Level (SAL)'].notna()].copy()
    key=['PWSID','location','date','Sample Number','analyte'];counts=d.groupby(key).value.nunique();bad=counts[counts>1]
    assert not len(bad),'Conflicting WA sample/analyte duplicate requires adjudication'
    x=d.drop_duplicates(key).pivot(index=['PWSID','location','date','Sample Number'],columns='analyte',values='value').reset_index()
    before=len(x);x=x.dropna(subset=['PFOA','PFOS'])
    x=x.groupby(['PWSID','location','date'],as_index=False).agg(PFOA=('PFOA','mean'),PFOS=('PFOS','mean'))
    alt=original[original.analyte.eq('PFOS')].drop_duplicates(['PWSID','location','date','Sample Number','value','State Action Level (SAL)']).groupby(['PWSID','location','date','Sample Number']).value.agg(['max','sum']).reset_index().groupby(['PWSID','location','date'])[['max','sum']].mean().rename(columns={'max':'PFOS_allmax','sum':'PFOS_allsum'}).reset_index()
    x=x.merge(alt,on=['PWSID','location','date'],validate='one_to_one')
    names=d[['local_id','PWSID','Water System Name']].drop_duplicates();names['in_current_small_inventory']=names.PWSID.isin(sm.PWSID);names['in_ucmr5_any_record']=names.PWSID.isin(excluded)
    names.to_csv(OUT/'wa_exact_id_crosswalk.csv',index=False)
    overlap=names.merge(ucmr.drop_duplicates('PWSID'),on='PWSID')
    overlap.to_csv(OUT/'wa_crosswalk_ucmr5_name_verification.csv',index=False)
    log['states']['WA']={'raw_pfoa_pfos_rows':len(original),'SAL_coded_pfoa_pfos_rows':len(d),'same_name_PFOS_rows_without_SAL':int(original['State Action Level (SAL)'].isna().sum()),'paired_samples':before,'paired_distinct_dates':len(x),'source_systems':d.PWSID.nunique(),'matched_ucmr5_exact_id_for_crosswalk_validation':len(overlap),'date_min':str(x.date.min()),'date_max':str(x.date.max()),'precision':'Public CSV displays integer ng/L only; no individual assay reporting limit or rounding rule. Published-value endpoint,4.0threshold and hypothetical±0.5quantization sensitivity are reported separately.','nd_reporting':'State rule WAC246-390-075 sets2.0ng/L SDRL forPFOA/PFOS andrequires estimatednumeric concentrations at/aboveSDRL whenMRLhigher. IndividualRLnotinCSV.'}
    return x

def build_schedules(events,state,frame=None,tag='strict'):
    frame=sm if frame is None else frame
    e=events.merge(frame[['PWSID','sw','P','f','u','src','pop','cell','PRIMACY_AGENCY_CODE']],on='PWSID',how='inner')
    e=e[(e.date>=START)&(e.date<=END)].copy();e=exclude_known_ids(e,excluded)
    e['need']=e.need_actual.astype(int) if 'need_actual' in e else np.where(e.sw.eq(1),4,2)
    rows=[];selected=[]
    for (pid,loc),g in e.groupby(['PWSID','location'],sort=True):
        k=int(g.need.iloc[0]);q=first_count_window(g,k,MAXSPAN)
        if q is None:continue
        # Uniform censoring matches the national comparison convention.
        raw=q[['PFOA','PFOS']].to_numpy(float);v=np.where(raw<4,0,raw)
        y=retained_mean_outcome(raw,4.05);y40=retained_mean_outcome(raw,4.)
        stop1=bool((v[0]==0).all());stop2=bool((v[:2]==0).all()) if k==4 else False
        row={'PWSID':pid,'location':loc,'state':state,'k':k,'span_days':(q.date.max()-q.date.min()).days,'y_full':y,'y_threshold4':y40,'y_R1':0 if stop1 else y,'y_R2':0 if stop2 else y,'y_R12':0 if (stop1 if k==2 else stop2) else y,'c_full':k,'c_R1':1 if stop1 else k,'c_R2':2 if stop2 else k,'c_R12':(1 if stop1 else k) if k==2 else (2 if stop2 else k)}
        if state=='WA':
            lo=np.maximum(raw-.5,0);hi=raw+.5
            lo=np.where(lo<4,0,lo);hi=np.where(hi<4,0,hi)
            row['y_quantization_lo']=int((lo.mean(axis=0)>=4.05).any());row['y_quantization_hi']=int((hi.mean(axis=0)>=4.05).any())
            for alt in ['PFOS_allmax','PFOS_allsum']:
                vv=q[['PFOA',alt]].to_numpy(float);vv=np.where(vv<4,0,vv)
                row['y_'+alt]=int((vv.mean(axis=0)>=4.05).any())
        rows.append(row)
        s=q.copy();s['sequence']=np.arange(1,k+1);s['PFOA_censored4']=v[:,0];s['PFOS_censored4']=v[:,1];selected.append(s)
    locations=pd.DataFrame(rows)
    locations.to_csv(OUT/(state+'_'+tag+'_locations.csv'),index=False)
    pd.concat(selected,ignore_index=True).to_csv(OUT/(state+'_'+tag+'_events.csv'),index=False)
    agg={c:'max' for c in ['y_full','y_threshold4','y_R1','y_R2','y_R12']};agg.update({c:'sum' for c in ['c_full','c_R1','c_R2','c_R12']});agg['location']='size'
    if state=='WA':agg.update({c:'max' for c in ['y_quantization_lo','y_quantization_hi','y_PFOS_allmax','y_PFOS_allsum']})
    s=locations.groupby('PWSID',as_index=False).agg(agg).rename(columns={'location':'n_locations'})
    s=s.merge(frame,on='PWSID',how='left');s['external_state']=state
    assert not set(s.PWSID)&excluded
    if len(s):
        assert (s.y_R12<=s.y_full).all() and (s.c_R12<=s.c_full).all()
    return s

def main():
    all_mean=[];strict=[]
    for state,loader in [('MI',mi_events),('WA',wa_events)]:
        events=loader();strict.append(build_schedules(events,state))
        # Full-data mean is a secondary endpoint, not schedule-harmonized.
        z=events.copy()
        for a in ['PFOA','PFOS']:z[a]=z[a].where(z[a]>=4,0)
        loc=z.groupby(['PWSID','location'])[['PFOA','PFOS']].mean();loc['y_all_mean']=(loc.max(axis=1)>=4.05).astype(int)
        systems=loc.groupby('PWSID').y_all_mean.max().reset_index().merge(sm,on='PWSID');systems=systems[~systems.PWSID.isin(excluded)];systems['external_state']=state;all_mean.append(systems)
    strict=pd.concat(strict,ignore_index=True);strict.to_csv(OUT/'external_schedule_systems.csv',index=False)
    pd.concat(all_mean,ignore_index=True).to_csv(OUT/'external_all_mean_systems.csv',index=False)
    for state,s in strict.groupby('external_state'):
        log['states'][state]['strict_n']=len(s);log['states'][state]['strict_positive']=int(s.y_full.sum());log['states'][state]['R12_missed']=int((s.y_full-s.y_R12).sum());log['states'][state]['c_full_sum']=int(s.c_full.sum());log['states'][state]['groundwater_n']=int(s.sw.eq(0).sum());log['states'][state]['groundwater_positive']=int(s.loc[s.sw.eq(0),'y_full'].sum())
    (OUT/'cohort_protocol_and_counts.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(log,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
