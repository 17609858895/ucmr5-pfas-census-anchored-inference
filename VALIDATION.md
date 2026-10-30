# Source-release validation

Validated on 2026-10-31 in the existing research analysis environment. The exact package versions used are recorded in `requirements-tested.txt`; `requirements.txt` gives compatible lower bounds for installation.

Five deterministic tests passed (`python -m unittest discover -s tests -v`):

1. Capped unequal inclusion probabilities retain their expected sample-size budget.
2. Inclusion probabilities calibrated with unequal schedule costs retain the expected analytical budget after caps.
3. Full observation and full-schedule completion reduce the two-phase estimator to the exact population-weighted mean with zero design variance.
4. Exhaustive enumeration of first- and second-phase selection in a three-unit population yields the known target exactly for the design-only two-phase estimator.
5. Scalar and vector interval routines agree, including prevalence boundary cases.

All 39 Python files passed AST parsing. Statistical function definitions were compared with the manuscript analysis source, including the difference estimator, tuning coefficient, variance, benchmark design rules, pseudo-population generation and two-phase implementation. The source mapping and hashes are in `source_manifest.json`.

No real-data model fitting, benchmark, bootstrap, simulation or publication-figure regeneration was executed for this source-packaging pass. These checks establish software identities and preserve source logic; they do not constitute a new scientific validation or a fresh full-pipeline reproduction.
