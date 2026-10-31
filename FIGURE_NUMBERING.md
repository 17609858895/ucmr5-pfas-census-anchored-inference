# Figure numbering

Plotting scripts keep their original output names so that the rebuild and audit steps stay reproducible. The submitted figures are numbered as follows.

| Script output | Submitted figure | Script |
|---|---|---|
| `Fig_1` | Fig. 1 | `plotting/generate_refined_figures.py` |
| `Fig_2` | Fig. 2 | `strengthening/spatial/render_spatial_figures.py` |
| `Fig3_anchor_method` | Fig. 3 (panel a drawn by `workflow_schematic.py` through `refine_mechanism_round3.workflow3`) | `plotting/generate_refined_figures.py` |
| `Fig_4` | Fig. 4 | `plotting/generate_refined_figures.py` |
| `Fig_5` | Fig. 5 | `render_upgraded_figures.py --only 5` |
| `Fig_6` | Fig. 6 | `strengthening/external/render_external_figures.py` |
| `Fig_7` | Fig. 7 | `render_upgraded_figures.py --only 7` (adds `strengthening/method/external_neyman_two_phase.py` results) |
| `Fig_S4` (spatial diagnostics) | Fig. S1 | `strengthening/spatial/render_spatial_figures.py` |
| `Fig_S2` | Fig. S2 | `plotting/generate_refined_figures.py` |
| `Fig_S1` (interval and completion sensitivities) | Fig. S3 | `plotting/generate_refined_figures.py` |
| `Fig_S3` (burden and cost scenarios) | Fig. S4 | `plotting/generate_refined_figures.py` |
| `Fig_S5` | Fig. S5 | `strengthening/external/render_external_figures.py` |
| `Fig_S6` | Fig. S6 | `strengthening/method/render_method_figures.py` |
| `Graphical_Abstract` | Graphical abstract | `render_upgraded_figures.py --only GA` |

The same-depth Neyman comparison in Fig. 7 and Table S15 is produced by

```
python -m strengthening.method.external_neyman_two_phase --cohort <external_schedule_systems.csv> --primary <external_primary_R12_B300.csv> --out <output folder>
```
