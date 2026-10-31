"""Executed, no-data helpers for independent cohort and schedule definitions."""
import re
import numpy as np
import pandas as pd

def washington_local_id(value):
    text=str(value).replace('="','').replace('"','').zfill(5)
    if not re.fullmatch(r'[A-Za-z0-9]{5}',text):raise ValueError('Invalid five-character Washington WSID')
    return text

def washington_pwsid(value):return 'WA53'+washington_local_id(value)

def effective_observation_end(all_state_dates,requested_end):
    """Follow-up opportunity cannot extend beyond actual state availability."""
    latest=pd.to_datetime(all_state_dates).max()
    if pd.isna(latest):raise ValueError('No observed state sampling date')
    return min(pd.Timestamp(requested_end),latest)

def source_event_count(source_type):
    if isinstance(source_type,str) and source_type.startswith('Ground Water'):return 2
    if source_type=='Surface':return 4
    return 0

def exclude_known_ids(frame,ids):
    """Exclude every known UCMR5 ID, regardless of complete label status."""
    return frame.loc[~frame.PWSID.isin(set(ids))].copy()

def first_count_window(frame,k,max_span=366):
    """First k distinct dates; reject, rather than search later, if ineligible."""
    q=frame.sort_values('date').head(int(k)).copy()
    if len(q)!=k or (q.date.max()-q.date.min()).days>max_span:return None
    return q

def first_complete_calendar(frame,years=(2023,2024,2025)):
    """First complete year and first paired observation in each period."""
    for year in years:
        h=frame[frame.date.dt.year.eq(year)].copy()
        if h.empty:continue
        k=int(h.need_actual.iloc[0])
        if k not in (2,4):raise ValueError('Calendar needs two GW or four surface periods')
        h['period']=((h.date.dt.month-1)//(6 if k==2 else 3)).astype(int)
        if set(h.period)!=set(range(k)):continue
        return h.sort_values('date').groupby('period',sort=True).head(1).sort_values('date')
    return None

def retained_mean_outcome(values,threshold=4.05):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('Paired finite PFOA/PFOS concentrations required')
    v=np.where(values<4,0,values)
    return int((v.mean(axis=0)>=threshold).any())
