# Additional analysis inputs and definitions

All paths can be passed as command-line arguments. Defaults use `ANCHOR_WORKDIR` and write to `results/review_sensitivity/`. Raw records and derived system-level files are intentionally absent from this source release.

| Input | Required columns and meaning |
|---|---|
| `data_derived/small_frame_anchor.parquet` | One row per inventory system; `PWSID`, `pop`, `P`, `src`, `scls`, `cell`, `y_mcl`, `f`, `u`, `EPA_REGION`, `PRIMACY_AGENCY_CODE`. `y_mcl` is missing for unobserved systems. `P` is the population-target weight; `f,u` are frozen larger-system predictions and error scores. |
| `data_derived/ucmr5_all.parquet` | `PWSID`, `Size`; original UCMR5 size-category crosswalk. `Size=S` is not an original <3,300-system sample-membership flag. |
| `data_derived/loc_event.parquet` | `PWSID`, `FacilityID`, `SamplePointID`, `WaterType`, `SampleEventCode`, datetime `date`, `v_PFOA`, `v_PFOS` in ng/L. |
| `data_derived/frame_systems.parquet` | `PWSID` for the matched active inventory. |
| `results/within_system_rules.parquet` | `PWSID`, `size_cat`, `y`, `cost_full`, `cost_R1`, `cost_R12`, `yt_R1`, `yt_R12`; retained schedule-replay outcomes and analytical counts. |
| `results/two_level_small.csv` | Frozen output of `s24_two_level.py`, including `scenario`, `seed`, `estimand`, `strategy`, `truth`, `mse`, `cov_kg`, `cost_mean`. |

`s25_weight_sensitivity.py` uses working cell weights `N_h/n_h`. Unsupported calibration margins are recorded and skipped, rather than assigning invented weights. The outcome-odds scenario leaves observed outcomes fixed and changes the assumed unobserved cell prevalences. Gamma one reproduces direct post-stratification. Bootstrap resampling is stratified by the eight observed cells; predictions and the inventory stay fixed. Canonical domain totals use one fitted national coefficient per target in every draw. The same resample supplies both targets, permitting paired share contrasts.

`s26_timing_sensitivity.py` excludes an entire retained location-event set if its last minus first event date exceeds 366 days, then recomputes system endpoints and working weights. It does not impose quarterly/semiannual spacing and does not refit predictions.

`s27_direct_budget_check.py` reconstructs the original pseudo-outcomes and donor-schedule assignments, verifies their truths, and reproduces the first 600 reference draws exactly. It then evaluates the shared R12 design with 20% random full completion. Both estimands use the same first/second-phase selections. Expected analytical counts, realized costs, MSE, coverage and Monte Carlo standard errors are saved by scenario and seed. The direct checks condition on the same fitted pseudo-populations; they are not external field validation. The summary command aggregates the dual designs over five seeds without grouping the reference by its slightly different floating-point budget values. Min/max whiskers span these five pseudo-populations, not confidence intervals.

The older independently fitted size-domain estimates and inverse-MSE cost projections remain separate outputs for reproducibility. Use `additive_domain_burden_canonical.csv` for additive composition shares, and `direct_budget_figure_data.csv` / `tableS11_direct_budget_canonical.csv` for directly tested budgets.
