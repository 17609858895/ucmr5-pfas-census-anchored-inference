"""Configurable source-only paths; importing paths does not run an analysis."""
from pathlib import Path
import os
REPO=Path(__file__).resolve().parents[1]
LEGACY_CODE=REPO/'legacy_support' if (REPO/'legacy_support').is_dir() else REPO
PLOTTING_CODE=LEGACY_CODE/'plotting'
ANALYSIS_CODE=LEGACY_CODE/'analysis'
def env_path(name,default):return Path(os.environ.get(name,default)).expanduser().resolve()
WORKDIR=env_path('ANCHOR_WORKDIR',REPO)
DERIVED=env_path('ANCHOR_DERIVED_DIR',WORKDIR/'data_derived')
EXTERNAL_RAW=env_path('ANCHOR_EXTERNAL_RAW_DIR',WORKDIR/'external_raw')
WA_RAW=env_path('ANCHOR_WA_RAW_DIR',EXTERNAL_RAW/'wa_doh_20261031')
RESULTS=env_path('ANCHOR_STRENGTHENING_RESULTS',WORKDIR/'results/strengthening')
EXTERNAL_RESULTS=env_path('ANCHOR_EXTERNAL_RESULTS',RESULTS/'external')
SPATIAL_RESULTS=env_path('ANCHOR_SPATIAL_RESULTS',RESULTS/'spatial')
METHOD_RESULTS=env_path('ANCHOR_METHOD_RESULTS',RESULTS/'method')
FIGURES=env_path('ANCHOR_STRENGTHENING_FIGURES',WORKDIR/'figures')

def prepare_outputs():
    for d in [EXTERNAL_RESULTS,SPATIAL_RESULTS,METHOD_RESULTS,FIGURES]:d.mkdir(parents=True,exist_ok=True)
