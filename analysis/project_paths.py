"""Portable paths for the source-only research release."""
from pathlib import Path
import os
PROJECT_ROOT=Path(os.environ.get("ANCHOR_WORKDIR",Path(__file__).resolve().parents[1])).expanduser().resolve()
RAW=Path(os.environ.get("ANCHOR_RAW_DIR",PROJECT_ROOT/"data"/"raw")).expanduser().resolve()
D=PROJECT_ROOT/"data_derived"
R=PROJECT_ROOT/"results"
UCMR5_ZIP=Path(os.environ.get("UCMR5_OCCURRENCE_ZIP",RAW/"ucmr5-occurrence-data.zip"))
SDWIS_ZIP=Path(os.environ.get("SDWIS_INVENTORY_ZIP",RAW/"SDWA_latest_downloads_20261015.zip"))
ZCTA_FILE=Path(os.environ.get("ZCTA_GAZETTEER_FILE",RAW/"2023_Gaz_zcta_national.txt"))
STATE_KML=Path(os.environ.get("STATE_BOUNDARY_KML",RAW/"cb_2024_us_state_20m.kml"))
N_JOBS=max(1,int(os.environ.get("ANCHOR_N_JOBS","1")))
D.mkdir(parents=True,exist_ok=True)
R.mkdir(parents=True,exist_ok=True)
