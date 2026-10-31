from strengthening import paths
from pathlib import Path
import pandas as pd
P=paths.EXTERNAL_RESULTS;A=P.parent;O=paths.EXTERNAL_RESULTS
s=pd.read_csv(O/'external_schedule_systems.csv')
for folder,stem,prefix in [('region_holdout','spatial_predictions_frame','cold'),('operational_coordinate_diagnostic','operational_coordinate_predictions','available'),('region_coordinate_diagnostic','spatial_predictions_frame','cold_diag')]:
    x=pd.read_parquet(paths.SPATIAL_RESULTS/folder/(stem+'.parquet'))
    cols=[c for c in ['attributes','geography','neighbours','coordinates','coordinates_neighbours'] if c in x]
    x=x[['PWSID']+cols].rename(columns={c:prefix+'_'+c for c in cols})
    s=s.merge(x,on='PWSID',validate='one_to_one')
assert not s[[c for c in s if c.startswith(('cold_','available_'))]].isna().any().any()
s.to_csv(O/'external_schedule_predictions.csv',index=False)
print(s.groupby('external_state').y_full.agg(['size','sum']).to_string())
