# Results

<!-- Drafted with Claude (Anthropic). Asked for (18 October 2026, section 5 of docs/build_plan.md): one line per file of this folder. Later asked for (20 October 2026, step 2 of section 10 of the build plan): the files of the checkpoint report. Later asked for (22 October 2026, step 6 of section 10 of the build plan): the files of stage 4. -->

Every file in this folder is written by a stage notebook in `code/` run in the Nuvolos workspace and committed from that run, except in `checkpoint/` (section 6 of the build plan): `checkpoint/inputs/` (and `checkpoint/rerun/inputs/`, if the checkpoint calls for a rerun) hold unchanged copies of the two files of part 1 of stage 3 committed from Nuvolos, and the report beside them is written by `03b_checkpoint.ipynb`, which fits nothing, run by Claude Code on those copies. Quick mode (`ADA_QUICK=1`) writes into `results_quick/` instead, which git ignores. Section 6 of [`docs/build_plan.md`](../docs/build_plan.md) describes each file in detail.

| File | Written by | What it holds |
|---|---|---|
| `cleaning_log.csv` | `02_clean.ipynb` | For each cleaning step and year: the rows before, dropped and left |
| `dev_runs.csv` | `03_development.ipynb` | One row per configuration and training draw in the development year: test and training deviance, the other scores, seconds |
| `dev_components.csv` | `03_development.ipynb` | One row per training size and configuration: the test error split into variance and remainder |
| `checkpoint/` | `03b_checkpoint.ipynb` | The checkpoint report on part 1 of stage 3. Its inputs, copied unchanged from part 1, are in `checkpoint/inputs/` |
| `size_rule.csv` | `04_size_rule.ipynb` | Per training size, one row for each of `ppml`, tuned `rf` and tuned `gbm`: the tuned settings, the mean, standard error and median of the test deviance, whether one draw holds more than half of the sum over draws (`se_unreliable`), whether the size rule chooses it, and the tuned settings at an edge of their grid |
| `gaps.csv` | `04_size_rule.ipynb` | Per training size and comparison (`ppml` against tuned `rf`, tuned `gbm` and the better of the two): the error gap with its paired standard error and whether `ppml` is within two standard errors, the bias and variance gaps in their three versions, and the draws in which `ppml` wins |
| `cutoff.csv` | `04_size_rule.ipynb` | Per comparison: the pattern of the error gap over the sizes, the development-year cutoff n* (or the last crossing), and the bootstrap shares of each pattern with the intervals |
| `analysis_runs.csv` | `05_analysis.ipynb` | One row per method and draw in the analysis year |
| `analysis_summary.csv` | `05_analysis.ipynb` | Per training size: the scores of each method, and the test of the size rule against cross-validation |
| `analysis_cutoff.csv` | `05_analysis.ipynb` | The analysis-year cutoff at the fixed settings: the test of H1 |
| `selection_confusion.csv` | `05_analysis.ipynb` | The model each selector picked against the model that won |
| `check_runs.csv`, `check_components.csv`, `check_cutoff.csv` | `06_checks.ipynb` | The same as stages 3 and 4 for the three checks: added noise, another test split, twice the trees |

TODO: the files appear here as their stages are run in Nuvolos.

## The checkpoint report

`03b_checkpoint.ipynb` writes these into `checkpoint/` (or `checkpoint/rerun/`), one to three for each item of the report in section 6 of the build plan. Its inputs are in `checkpoint/inputs/`.

| File | Item | What it holds |
|---|---|---|
| `learning_curves.pdf`, `learning_curves.csv` | 1 | Mean test and training deviance of `ppml`, tuned `rf`, tuned `gbm`, `mean` and `tree` by size, with bands of two standard errors |
| `scores.csv` | 2 | For the same five lines: mean, standard error, median and largest draw of the test deviance, D squared, MAE and RMSE |
| `selection_counts.csv`, `selection_scores.csv`, `ppml_wins.csv` | 3 | The model the size rule picks against the model that won each draw, the share picked right and the regret, and the draws in which `ppml` beats tuned `rf`, tuned `gbm` and the better of the two |
| `gaps.csv`, `crossing.csv` | 4 | Error, bias and variance gaps of `ppml` against each ensemble, and the provisional pattern of each comparison |
| `tuned.csv`, `edges.csv` | 5 | The tuned settings by size, and whether each sits at an edge of its grid |
| `groups.csv` | 6 | The test deviance of each line split over the largest 1% of test flows, the next 9% and the rest |
| `health.csv`, `worse_than_mean.csv`, `divergences.csv` | 7 | By size: the identity error = remainder_raw + variance_raw of section 8 (the other identity, behind `gaps.csv`, is asserted), the constant columns and fallbacks of `ppml`; then the draws worse than `mean`, and the diverging `gbm` draws |
| `time.csv`, `time_projection.csv` | 8 | Seconds per draw by size, and the projected minutes of the full run of stage 3 and of stages 5 and 6 |
| `coefficients.csv`, `thin_indicators.csv` | 9 | The spread of the `ppml` coefficients over draws, and the draws in which an indicator had few 1s |
| `rules.csv` | | What each rule of the checkpoint calls for |
