# Figures

<!-- Drafted with Claude (Anthropic). Asked for: a note stating the rule for this folder. Later asked for (18 October 2026, section 5 of docs/build_plan.md): the files stage 7 will write. Later asked for (21 October 2026, decision 19 of the build plan): the checkpoint report on the rerun of part 1. Later asked for (21 October 2026, decision 20 of the build plan): the ridge line in the learning curves. Later asked for (23 October 2026, stage 7 of the build plan): one line per file that stage 7 writes. Later asked for (24 October 2026, decision 23 of the build plan): the tables of the models and of the variables. -->

Every file in this folder is written by `code/07_figures.ipynb` (Requirement 4.11). Nothing is added by hand. Figures are saved as `.pdf`, tables as `.csv` and `.tex` (decision 7 of `docs/build_plan.md`).

`07_figures.ipynb` writes every file below from the files of stages 2 to 6 in `results/`, from `data/gravity_extract.csv` and `data/gravity_clean.csv`, and, for `tab_models`, from the settings in `code/helpers.py`, after checking that those files agree with each other (section 6 of the build plan, stage 7). Each table is a `.csv` with the full numbers and a `.tex` with the same rows rounded for the paper: a LaTeX `tabular` with the rules of the `booktabs` package, to be put in a table with its caption. Each figure is a `.pdf`, with a `.csv` of the numbers it draws. Captions are Yu's and go in the paper.

| File | What it shows |
|---|---|
| `tab_models` | The settings of `ppml`, `rf` and `gbm`: the scikit-learn estimator, the settings tuned with the values tried, and the fixed settings, read from `code/helpers.py` |
| `tab_variables` | The columns of the CEPII extract, the unit of each and what stage 2 does with it, checked on `data/gravity_extract.csv` and `data/gravity_clean.csv` |
| `tab_cleaning` | The rows of each year in the extract and after each cleaning step |
| `tab_summary` | Summary statistics of the target and the eight predictors in the cleaned table, per year |
| `tab_tuned` | The tuned settings of the development year and the test errors by training size, with the model the size rule picks and the settings at an edge of their grid (`results/size_rule.csv`) |
| `tab_cutoff` | The pattern of the error gap, the reversal, the cutoff with its interval, and the share of each pattern over the bootstrap resamples, for every comparison of the development year (base run, `noise`, `split2`) and of the analysis year (the test of H1, and the cutoff test of `ridge` beside it) |
| `tab_selection` | The size rule against cross-validated selection in the analysis year, at each size: the test of H2, and the fits and seconds per draw |
| `tab_selection_confusion` | The model each selector picked against the model that won, over all sizes of the analysis year |
| `tab_checks` | `ppml` against the better ensemble at each size in the base run and in the checks `noise` and `split2` of stage 6: error gap, draws won, bias gap and variance gap |
| `tab_trees2x` | The check `trees2x` of stage 6: the tuned ensembles with more trees against the tuned ones, at the largest size |
| `learning_curves` | Test and training deviance by training size in the development year, for `ppml`, tuned `rf`, tuned `gbm`, `mean`, `tree`, and `ridge`, added at the checkpoint (decision 20) |
| `components` | Variance and remainder of `ppml`, tuned `rf` and tuned `gbm` by training size in the development year |
| `gaps` | The error, bias and variance gaps of `ppml` against each ensemble in the development year, as shares of the ensemble's error, with the cutoff and its interval |
| `analysis_methods` | The three models at the frozen settings, the size rule, cross-validated selection and the best of the three in the analysis year |
| `noise_gaps` | The gaps of `gaps` with and without added noise in the training flows (stage 6, `noise`) |

The `.csv` beside a figure has the same name as the figure, so `figures/gaps.csv` holds the numbers of `gaps.pdf`; it is not the file `results/gaps.csv` of stage 4, which it is computed from.

`code/03b_checkpoint.ipynb`, run at the checkpoint, writes its learning curves to `results/checkpoint/` (and, for the rerun of part 1, to `results/checkpoint/rerun/`), not here: they are a report on part 1 of stage 3, not a figure of the paper.
