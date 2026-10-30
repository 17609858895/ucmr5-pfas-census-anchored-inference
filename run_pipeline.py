"""Explicit stage runner; no stage runs merely by importing this module."""
from pathlib import Path
import argparse,os,subprocess,sys
REPO=Path(__file__).resolve().parent
STAGES={
 'load':'s00_load.py','events':'s01_build_tables.py','covariates':'s03_covariates.py',
 'frame':'s07_frame.py','features':'s08_build_features.py','benchmark':'s17_census_grid.py',
 'benchmark-summary':'s18_summarize_grid.py','burden':'s19_small_frame_burden.py',
 'projection':'s20_small_projection.py','external':'s21_external_states.py',
 'recalibration':'s22_external_followup.py','schedules':'s23_within_system.py',
 'two-phase':'s24_two_level.py','figure-data':'f00_figure_data.py'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=[*STAGES,'figures','check-inputs'])
 ap.add_argument('extra',nargs=argparse.REMAINDER);a=ap.parse_args()
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
 env.setdefault('ANCHOR_WORKDIR',str(REPO));env.setdefault('ANCHOR_N_JOBS','1')
 if a.stage=='check-inputs':
  sys.path.insert(0,str(REPO/'analysis'));import project_paths as p
  files=[p.UCMR5_ZIP,p.SDWIS_ZIP,p.ZCTA_FILE]
  for f in files:print(('OK ' if f.is_file() else 'MISSING ')+str(f))
  raise SystemExit(0 if all(f.is_file() for f in files) else 1)
 script=REPO/('plotting/generate_refined_figures.py' if a.stage=='figures' else 'analysis/'+STAGES[a.stage])
 subprocess.run([sys.executable,str(script),*a.extra],env=env,cwd=REPO,check=True)
