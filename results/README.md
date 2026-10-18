# Results

<!-- Drafted with Claude (Anthropic). Asked for (18 October 2026, section 5 of docs/build_plan.md): one line per file of this folder. -->

Every file in this folder is written by a stage notebook in `code/` and committed from a full run in the Nuvolos workspace. Quick mode (`ADA_QUICK=1`) writes into `results_quick/` instead, which git ignores. Section 6 of [`docs/build_plan.md`](../docs/build_plan.md) describes each file in detail.

| File | Written by | What it holds |
|---|---|---|
| `cleaning_log.csv` | `02_clean.ipynb` | For each cleaning step and year: the rows before, dropped and left |
| `dev_runs.csv` | `03_development.ipynb` | One row per configuration and training draw in the development year: test and training deviance, the other scores, seconds |
| `dev_components.csv` | `03_development.ipynb` | One row per training size and configuration: the test error split into variance and remainder |
| `checkpoint/` | `03b_checkpoint.ipynb` | The checkpoint report on part 1 of stage 3, with its inputs in `checkpoint/inputs/` |
| `size_rule.csv` | `04_size_rule.ipynb` | Per training size: the tuned setting and test error of each model, and the model the size rule chooses |
| `gaps.csv` | `04_size_rule.ipynb` | Error, bias and variance gaps of `ppml` against each ensemble, by training size |
| `cutoff.csv` | `04_size_rule.ipynb` | The development-year cutoff n* of each comparison, with its bootstrap interval |
| `analysis_runs.csv` | `05_analysis.ipynb` | One row per method and draw in the analysis year |
| `analysis_summary.csv` | `05_analysis.ipynb` | Per training size: the scores of each method, and the test of the size rule against cross-validation |
| `analysis_cutoff.csv` | `05_analysis.ipynb` | The analysis-year cutoff at the fixed settings: the test of H1 |
| `selection_confusion.csv` | `05_analysis.ipynb` | The model each selector picked against the model that won |
| `check_runs.csv`, `check_components.csv`, `check_cutoff.csv` | `06_checks.ipynb` | The same as stages 3 and 4 for the three checks: added noise, another test split, twice the trees |

TODO: the files appear here as their stages are run in Nuvolos.
