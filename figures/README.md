# Figures

<!-- Drafted with Claude (Anthropic). Asked for: a note stating the rule for this folder. Later asked for (18 October 2026, section 5 of docs/build_plan.md): the files stage 7 will write. -->

Every file in this folder is written by `code/07_figures.ipynb` (Requirement 4.11). Nothing is added by hand. Figures are saved as `.pdf`, tables as `.csv` and `.tex` (decision 7 of `docs/build_plan.md`).

TODO, with stage 7: one line per file. The plan (section 6, stage 7) lists:

| File | What it shows |
|---|---|
| `tab_cleaning` | Rows after each cleaning step, per year |
| `tab_summary` | Summary statistics of the cleaned table, per year |
| `tab_tuned` | Tuned settings and test errors by training size |
| `tab_cutoff` | The cutoff, its interval and the share of each pattern, for every comparison and variant |
| `tab_selection` | The size rule against cross-validated selection in the analysis year |
| `tab_selection_confusion` | The model each selector picked against the model that won, over all sizes |
| `tab_checks` | The checks of stage 6 |
| `learning_curves` | Test and training deviance of the five models by training size |
| `components` | Variance and remainder of the three models by training size |
| `gaps` | Bias gap and variance gap, with the cutoff and its interval |
| `analysis_methods` | The three models, the size rule, cross-validated selection and the best of the three in the analysis year |
| `noise_gaps` | The gaps with and without added noise |

`code/03b_checkpoint.ipynb`, run once at the checkpoint, writes its learning curves to `results/checkpoint/`, not here: they are a report on part 1 of stage 3, not a figure of the paper.
