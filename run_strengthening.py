"""Run new validation stages without changing the historical pipeline."""
from pathlib import Path
import argparse,os,subprocess,sys
REPO=Path(__file__).resolve().parent
STAGES={
 'spatial':'spatial.spatial_transfer',
 'spatial-coordinates':'spatial.spatial_coordinate_diagnostic',
 'available-coordinates':'spatial.operational_coordinate_diagnostic',
 'outside-region-calibration':'spatial.calibration_by_outer_region',
 'spatial-summary':'spatial.summarize_spatial',
 'information-regimes':'spatial.compare_information_regimes',
 'buffer-comparison':'spatial.compare_buffer',
 'validate-spatial':'spatial.validate_spatial',
 'burden-model-stress':'spatial.burden_model_stress',
 'figures-spatial':'spatial.render_spatial_figures',
 'audit-spatial-figures':'spatial.audit_spatial_figures',
 'external-counts':'external.build_external_cohorts',
 'external-source-verify':'external.verify_wa_source',
 'external-predictions':'external.prepare_external_predictions',
 'external-metrics':'external.external_metrics',
 'external-calibration':'external.calibration_transfer',
 'external-followup':'external.followup_selection_diagnostic',
 'external-calendar':'external.calendar_sensitivity',
 'external-sensitivity':'external.additional_external_sensitivity',
 'external-secondary':'external.secondary_nonharmonized_states',
 'external-freeze':'external.freeze_external_protocol',
 'figures-external':'external.render_external_figures',
 'completion-pseudo':'method.completion_design_validation',
 'completion-external':'method.run_external_v2',
 'completion-range-newseeds':'method.range_new_pseudo_validation',
 'completion-calendar':'method.calendar_external_sensitivity',
 'completion-audit':'method.audit_estimators',
 'completion-summary':'method.finalize_method_handoff',
 'figures-method':'method.render_method_figures',
}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('stage',choices=[*STAGES,'check-inputs','test','install-reference','join-spatial-diagnostics'])
 ap.add_argument('--force-reference',action='store_true',help='Explicitly replace existing different aggregate reference inputs')
 for option in ['workdir','derived-dir','external-raw-dir','wa-raw-dir','results-dir','external-results','spatial-results','method-results','figure-dir']:ap.add_argument('--'+option,type=Path)
 args,extra=ap.parse_known_args();env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
 keys={'workdir':'ANCHOR_WORKDIR','derived_dir':'ANCHOR_DERIVED_DIR','external_raw_dir':'ANCHOR_EXTERNAL_RAW_DIR','wa_raw_dir':'ANCHOR_WA_RAW_DIR','results_dir':'ANCHOR_STRENGTHENING_RESULTS','external_results':'ANCHOR_EXTERNAL_RESULTS','spatial_results':'ANCHOR_SPATIAL_RESULTS','method_results':'ANCHOR_METHOD_RESULTS','figure_dir':'ANCHOR_STRENGTHENING_FIGURES'}
 for field,key in keys.items():
  value=getattr(args,field)
  if value is not None:env[key]=str(value.expanduser().resolve())
 env.setdefault('ANCHOR_WORKDIR',str(REPO));env.setdefault('ANCHOR_N_JOBS','1')
 if args.stage=='test':
  command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_strengthening*.py','-v']
 elif args.stage in ['check-inputs','install-reference','join-spatial-diagnostics']:
  os.environ.update({key:env[key] for key in keys.values() if key in env})
  from strengthening import paths
  if args.stage=='join-spatial-diagnostics':
   import pandas as pd
   target=paths.SPATIAL_RESULTS/'region_holdout'
   a=pd.read_parquet(target/'spatial_predictions_frame.parquet')
   b=pd.read_parquet(paths.SPATIAL_RESULTS/'region_coordinate_diagnostic/spatial_predictions_frame.parquet')
   columns=['PWSID','coordinates','coordinates_neighbours']
   if any(c in a for c in columns[1:]):raise ValueError('Diagnostic columns already present in original frame')
   result=a.merge(b[columns],on='PWSID',validate='one_to_one')
   assert len(result)==len(a)
   result.to_parquet(target/'spatial_predictions_with_diagnostic.parquet',index=False)
   print('Joined coordinate diagnostic by exact PWSID; original frame unchanged.')
   return 0
  if args.stage=='install-reference':
   import shutil,hashlib
   paths.prepare_outputs()
   for group,target in [('external',paths.EXTERNAL_RESULTS),('spatial',paths.SPATIAL_RESULTS),('method',paths.METHOD_RESULTS),('figure_data',paths.WORKDIR/'figure_data')]:
    source=REPO/'strengthening/reference_results'/group
    if not source.exists():continue
    for f in source.rglob('*'):
     if not f.is_file():continue
     destination=target/f.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
     if destination.exists() and destination.read_bytes()!=f.read_bytes() and not args.force_reference:raise RuntimeError('Existing different output; use a fresh workdir or --force-reference: '+str(destination))
     shutil.copy2(f,destination)
   print('Installed aggregate reported-result inputs for plotting; no model fitting or empirical validation was rerun.')
   return 0
  files=[paths.DERIVED/name for name in ['frame_systems.parquet','small_frame_anchor.parquet','ucmr5_all.parquet']]+[paths.WA_RAW/name for name in ['PFAS_Downloadable_Data.csv','PFAS_Data_Dictionary.csv','wa_general.xlsx','wa_sources.xlsx']]
  for f in files:print(('OK ' if f.is_file() else 'MISSING ')+str(f))
  return 0 if all(f.is_file() for f in files) else 1
 else:
  os.environ.update({key:env[key] for key in keys.values() if key in env})
  from strengthening import paths
  paths.prepare_outputs()
  command=[sys.executable,'-m','strengthening.'+STAGES[args.stage],*extra]
 return subprocess.run(command,cwd=REPO,env=env,check=False).returncode

if __name__=='__main__':raise SystemExit(main())
