# Source-release validation

Validated on 2026-10-31 in the research analysis environment. Exact tested package versions are in `requirements-tested.txt`.

Nine deterministic/synthetic tests passed (`python -m unittest discover -s tests -v`). They check capped expected sample size/cost, complete-observation identities, exact two-phase design expectation on an enumerated three-unit population, scalar/vector interval agreement, calibration margins, gamma-one post-stratification, additive domain totals and the 366/367-day eligibility boundary.

Existing statistical function ASTs and nine additional copied numerical functions agree with their analysis sources. Only portable paths, command-line interfaces, output organization, documentation labels and figure presentation were adapted. The full analysis/model-training pipeline was not rerun.

The added weight/bootstrap, timing and direct-budget commands were executed against frozen study inputs. Their CSV outputs agree with the reviewed analyses within floating-point precision. Direct-budget reproduction verified all ten original pseudo-population truths and all twenty original 600-draw reference MSE/coverage/cost results, before evaluating 1,000 draws per setting at 1,000/1,200 expected analytical samples. Source-file hashes remained unchanged during the successful verification run.

The portable current plotting entry point was smoke-tested with Figure 5 in an isolated workspace and visually inspected. Its palette and overlap audits reported zero errors. Publication-figure export and visual QA are maintained in the manuscript package. Those image files and the optional Figure 3a illustrative raster are not included in this source-only repository.

These checks establish software identities and reproduce the conditional analysis. They do not recover unknown official inclusion probabilities, make the pseudo-populations independent validation data, or establish field operational costs. Reproducibility details and file hashes are in `validation.json` and `source_manifest.json`.
