# Census-anchored PFAS burden and monitoring inference

Core analysis and publication-plotting source for **Census-anchored inference of PFAS burden and monitoring needs in small U.S. water systems**.

The code links the final UCMR5 monitoring records to a public-water-system inventory, trains source-population risk/error models, estimates weighted small-system burden, compares survey designs and estimators, and evaluates reduced schedules with random full-schedule completion. The two estimands are the share of systems and the share of service population attached to exceeding systems.

## Release contents

- `analysis/anchor_core.py`: capped sampling probabilities, stratified/Neyman allocation, Hájek/difference/GREG estimators and interval families.
- `analysis/s17_census_grid.py`: cross-size finite-population benchmark and ablations. Internal source folds use the fixed census-neighbour layer; target outcomes are excluded from cross-size training and neighbour features.
- `analysis/s19_small_frame_burden.py`: five-seed national prior and reconstructed-cell weighted burden estimation, including systems without coordinates.
- `analysis/s20_small_projection.py`: fitted, AUC-matched and extra-variation pseudo-populations on the real small-system frame.
- `analysis/s21_external_states.py` and `s22_external_followup.py`: external endpoints, discrimination and nationally fitted recalibration.
- `analysis/s23_within_system.py` and `s24_two_level.py`: schedule replay, analytical-budget calibration, two-phase correction/variance, shared dual allocation and projected equal-MSE requirements.
- `analysis/f00_figure_data.py` and `plotting/`: figure-table preparation and the current three-row main figures. Independent panels are redrawn. Figure 3 contains no estimator equation; that equation is in the manuscript Methods.

No manuscript, raw monitoring download, derived system-level dataset, fitted model, credentials or private absolute path is included. The illustrative raster used in Figure 3a is not redistributed; its plotting builder uses a simple editable scaffold if that optional asset is absent. Statistical panel values and uncertainty definitions use the retained analysis rules.

## Inputs and setup

Python 3.11/3.12 is recommended. Install `requirements.txt` in a dedicated environment. The source-only checks were run in the research analysis environment; the complete computational pipeline was not rerun for this packaging release.

Place the versioned public inputs listed in [DATA_SOURCES.md](DATA_SOURCES.md) under `data/raw/`, or set these environment variables:

| Variable | Default |
|---|---|
| `ANCHOR_WORKDIR` | repository root |
| `ANCHOR_RAW_DIR` | `<workdir>/data/raw` |
| `UCMR5_OCCURRENCE_ZIP` | `ucmr5-occurrence-data.zip` under raw |
| `SDWIS_INVENTORY_ZIP` | `SDWA_latest_downloads_20261015.zip` under raw |
| `ZCTA_GAZETTEER_FILE` | `2023_Gaz_zcta_national.txt` under raw |
| `STATE_BOUNDARY_KML` | `cb_2024_us_state_20m.kml` under raw |
| `ANCHOR_N_JOBS` | `1` parallel simulation worker |

For the state checks, retain the three exact input paths listed in `DATA_SOURCES.md` under `<workdir>/external_raw/`. Administrative and geographic inputs must use the study snapshots and field schemas. The current official URLs can serve newer versions: the occurrence ZIP hash identifies the study input and should be checked before claiming identical results.

## Execution order

```bash
python run_pipeline.py check-inputs
python run_pipeline.py load
python run_pipeline.py events
python run_pipeline.py covariates
python run_pipeline.py frame
python run_pipeline.py features
python run_pipeline.py benchmark
python run_pipeline.py benchmark-summary
python run_pipeline.py burden
python run_pipeline.py projection
python run_pipeline.py external
python run_pipeline.py recalibration
python run_pipeline.py schedules
python run_pipeline.py two-phase
python run_pipeline.py figure-data
python run_pipeline.py figures
```

The benchmark and two-phase stages can require substantial computation. The benchmark accepts `--quick`, `--cheap-only` and `--ml-only`; quick outputs are labelled separately and do not reproduce the manuscript configurations. Cached population objects are stored locally in `data_derived/sim_pops/`; only load caches you created yourself.

Before a full run, the small deterministic checks require no real input data:

```bash
python -m unittest discover -s tests -v
```

These checks cover expected sampling cost, complete-data identities, exact design-unbiasedness on a tiny enumerated population, and agreement of scalar/vector interval routines. They are software checks, not a new empirical validation of the study conclusions.

## Interpretation

The analytical endpoint is any retained complete-location mean PFOA or PFOS concentration at least 4.05 ng/L, with below-reporting-level values set to zero for this averaging rule. It is not a regulatory compliance determination. Original small-system unit inclusion probabilities were unavailable; national inference is conditional on the reconstructed eight cells. AUC for the national small sample is developmental. State endpoints and periods differ. Shared-design equal-MSE costs are inverse-MSE projections; interval coverage was evaluated at the simulated 1,500/3,000-sample budgets, not newly simulated at each projected cost.

No licence is assigned by this release. Please contact the author regarding reuse permissions. Source preparation and checks are documented in `source_manifest.json` and `VALIDATION.md`.
