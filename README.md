# Hidden PFOA and PFOS burden in small U.S. drinking-water systems: estimates and monitoring design

The **31 October 2026 reviewer revision** adds a same-depth classical control and the final figure set. `strengthening/method/external_neyman_two_phase.py` combines target-specific five-stratum Neyman allocation with R12 schedules and 20% random full-schedule completion in the Michigan and Washington cohorts; its aggregate output is `strengthening/reference_results/method/external_neyman_two_phase.csv`. `render_upgraded_figures.py` draws the final Fig. 5, Fig. 7 (nine designs grouped by sampling depth) and the graphical abstract from aggregate inputs, and `workflow_schematic.py` draws the vector workflow used in Fig. 3a and the graphical abstract. Model labels in the spatial, external and method renderers read "National model". Script output names and submitted figure numbers are mapped in [FIGURE_NUMBERING.md](FIGURE_NUMBERING.md); checks are recorded in `strengthening/REVIEWER_REVISION_VALIDATION.json`.

The **31 October 2026 strengthening release** adds fully UCMR5-disjoint Michigan/Washington cohorts, EPA-region information exclusions, schedule/precision/selection diagnostics and variable-probability completion analyses. Use [STRENGTHENING.md](STRENGTHENING.md) and `run_strengthening.py` for the current added analyses and figures. The earlier `s21`/`s22` state checks below are retained historical routines; their mixed endpoint/cohort comparisons are superseded by the new source-verified protocol. The numerical analysis files remain unchanged.

The subsequent figure-only revision uses **normal weight for all figure text**, including tick labels, axis labels, panel letters, legends and annotations. Font sizes, `nature_new` colours, numerical inputs and three-row main-figure layouts are retained. Current renderers and their publication helpers include this preference, so regeneration keeps the regular typography. The later compact-layout update reduces panel gaps, legend-strip space and canvas heights while keeping the font sizes, normal weight, colours, numerical inputs and three content rows. Its source and export checks are recorded separately in `strengthening/COMPACT_LAYOUT_VALIDATION.json`. A narrow follow-up adjusts the internal Fig.1a map-to-zoom whitespace and the main Fig.7 column gutter identified in the author's screenshots; the remaining figure layouts and all statistical inputs are retained. See `strengthening/TARGETED_WHITESPACE_VALIDATION.json`. The checks for this revision are recorded separately in `strengthening/REGULAR_WEIGHT_VALIDATION.json`; previous validation reports describe their original releases.

Core analysis and publication-plotting source for **Hidden PFOA and PFOS burden in small U.S. drinking-water systems: estimates and monitoring design**.

The code links the August 2026 UCMR5 occurrence-data snapshot to a public-water-system inventory, trains larger-system prediction/error models, estimates weighted small-system burden, compares survey designs and estimators, and evaluates reduced schedules with random full-schedule completion. The two estimands are the share of systems and the share of service population attached to exceeding systems.

## Release contents

- `analysis/anchor_core.py`: capped sampling probabilities, stratified/Neyman allocation, Hájek/difference/GREG estimators and interval families.
- `analysis/s17_census_grid.py`: cross-size finite-population benchmark and ablations. Internal source folds use the fixed census-neighbour layer; target outcomes are excluded from cross-size training and neighbour features.
- `analysis/s19_small_frame_burden.py`: five-seed national prediction and working source-by-size post-stratified burden estimation, including systems without coordinates.
- `analysis/s20_small_projection.py`: fitted, AUC-matched and extra-variation pseudo-populations on the real small-system frame.
- `analysis/s21_external_states.py` and `s22_external_followup.py`: external endpoints, discrimination and nationally fitted recalibration.
- `analysis/s23_within_system.py` and `s24_two_level.py`: schedule replay, analytical-budget calibration, two-phase correction/variance, shared dual allocation and projected equal-MSE requirements.
- `analysis/s25_weight_sensitivity.py`: alternative weights, observable-margin raking, outcome-odds scenarios and additive size-domain estimates with a common nationally fitted coefficient per target/bootstrap draw.
- `analysis/s26_timing_sensitivity.py`: whole-location exclusion when the retained event-set span exceeds 366 days; predictions remain fixed.
- `analysis/s27_direct_budget_check.py` and `s28_summarize_direct_budget.py`: directly evaluated 1,000/1,200 expected analytical-sample budgets, original-reference reproduction checks and correctly grouped scenario summaries.
- `analysis/f00_figure_data.py` and `plotting/`: figure-table preparation and the current three-row main figures. Independent panels are redrawn. Figure 3 contains no estimator equation; that equation is in the manuscript Methods.

No manuscript, raw monitoring download, derived system-level dataset, fitted model, credentials or private absolute path is included. Figure 3a is drawn as vector graphics by `workflow_schematic.py`; no raster illustration is used. Statistical panel values and uncertainty definitions use the retained analysis rules.

## Inputs and setup

Python 3.11/3.12 is recommended. Install `requirements.txt` in a dedicated environment. The complete computational pipeline was not rerun for this packaging release. The added weight/bootstrap, timing and direct-budget commands were rerun against the frozen study inputs and checked against the reviewed analysis outputs; see `VALIDATION.md`.

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
python run_pipeline.py weight-sensitivity
python run_pipeline.py timing-sensitivity
python run_pipeline.py direct-budget
python run_pipeline.py direct-budget-summary
python run_pipeline.py figure-data
python run_pipeline.py figures
```

The benchmark and two-phase stages can require substantial computation. The benchmark accepts `--quick`, `--cheap-only` and `--ml-only`; quick outputs are labelled separately and do not reproduce the manuscript configurations. Cached population objects are stored locally in `data_derived/sim_pops/`; only load caches you created yourself.

Before a full run, the small deterministic checks require no real input data:

```bash
python -m unittest discover -s tests -v
```

These checks cover expected sampling cost, complete-data identities, exact design-unbiasedness on a tiny enumerated population, agreement of scalar/vector interval routines, additive domain totals, the outcome-odds baseline, calibration margins and the 366/367-day eligibility boundary. They are software checks, not an independent empirical validation of the study conclusions.

The additional commands accept explicit input/output paths (`--help`). Their default outputs are under `results/review_sensitivity/`. Direct-budget evaluation requires the frozen `results/two_level_small.csv` and reproduces its original 600 reference draws before extending to 1,000. `--original-draws` must match the reference file. `--draws`, `--seeds`, bootstrap `--replicates` and budgets can be changed for exploratory runs; only defaults reproduce the reported configuration. Input schemas are specified in [REVIEW_ANALYSES.md](REVIEW_ANALYSES.md).

## Interpretation

The analytical endpoint is any retained location mean PFOA or PFOS concentration at least 4.05 ng/L, with below-reporting-level values set to zero for this averaging rule. Primary eligibility requires the reported two/four-event sets, without enforcing calendar spacing. It is not a regulatory compliance determination. Original small-system unit inclusion probabilities and original sample membership were unavailable; current-cohort estimates and intervals condition on eight working post-stratification cells and within-cell exchangeability. Historical `EPA`/`reconstructed` labels in older source routines denote the observed-cell reference allocation, not recovery of the official probability-sampling design. Prospective simulation designs have known inclusion probabilities.

Current inventory size classification uses ≤3,300 people; the original UCMR5 small-system sampling boundary was <3,300. The added boundary exclusion is a sensitivity check. AUC for the observed small-system cohort is developmental; state endpoints and periods differ. Common-national-coefficient domain totals in `additive_domain_burden_canonical.csv` are additive; independently fitted domain outputs from the earlier estimator are retained separately and are not the canonical composition shares.

Outcome-odds scenarios specify assumed exceedance odds in unmonitored relative to observed systems within cells. They are not confidence intervals or estimated selection odds. Older equal-MSE analytical costs are inverse-MSE projections. The added direct-budget evaluations simulate 1,000/1,200 analytical samples in the same fitted/AUC-matched pseudo-populations. At 1,200 samples they show approximately 57% expected sample reduction and population-target MSE ratios of 1.02/0.83, while system-target interval coverage is 0.938/0.930. Sample efficiency therefore does not imply nominal coverage. Travel, staffing and other operational costs are not modeled.

No licence is assigned by this release. Please contact the author regarding reuse permissions. Source preparation and checks are documented in `source_manifest.json` and `VALIDATION.md`.
