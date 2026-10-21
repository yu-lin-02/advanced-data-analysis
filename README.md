# Semester project: Advanced Data Analysis, HEC Lausanne, autumn 2026

<!-- Outline drafted with Claude (Anthropic). Asked for: a README skeleton in the order given in the course document "Starting and Shipping Your Project". Later asked for (18 October 2026, section 5 of docs/build_plan.md): the seven stages of the build plan and their files. Later asked for (20 October 2026, step 2 of section 10 of the build plan): how to run the checkpoint report. Later asked for (20 October 2026, step 4 of section 10 of the build plan): the date the checkpoint report was run. -->

## What this project does

It asks when a complex model loses to a simpler one in predicting bilateral trade: below which training size the gravity equation, estimated by Poisson pseudo-maximum likelihood, predicts trade flows better than a random forest or gradient-boosted trees, and whether choosing the model by training size alone does as well as cross-validation at a lower computing cost. The code cleans the CEPII Gravity data for 2017 and 2019, fits the three models on training samples of 50 to 5,000 country pairs, and writes the test errors, their split into bias and variance, and the figures and tables of the paper.

## Install

Python 3.12, the version in the course Nuvolos workspace. From the repository folder:

```
pip install -r requirements.txt
```

## Run

One command runs the stages in order, each notebook top to bottom in a fresh kernel:

```
python code/run_all.py
```

Other ways to run it:

| Command | What it runs |
|---|---|
| `python code/run_all.py --download` | Stage 1 first, then stages 2 to 7 |
| `python code/run_all.py --only 02_clean.ipynb` | One stage only |
| `python code/run_all.py --part 1` | Part 1 of stage 3 only: the training sizes up to 1,000, before the checkpoint |
| `python code/run_all.py --only 03b_checkpoint.ipynb` | The checkpoint report, on the copies of the two files of part 1 in `results/checkpoint/inputs/`. With `ADA_CHECKPOINT=rerun` in front, on the copies of a rerun of part 1 in `results/checkpoint/rerun/inputs/` instead |
| `ADA_QUICK=1 python code/run_all.py` | A quick test that the code runs, with few sizes and draws. Its results and figures go to `results_quick/` and `figures_quick/`, which git ignores, and are not evidence. Stage 2 has no quick-mode setting, so it writes the same `data/gravity_clean.csv` as a full run: never commit that file from a run outside the Nuvolos workspace |
| `ADA_JOBS=4 python code/run_all.py` | Four workers for the long stages. The results do not depend on the number of workers |

| Order | File | What it does |
|---|---|---|
| 1 | `code/01_get_data.ipynb` | Downloads the CEPII Gravity file into `data/raw/` (not committed), checks it, and saves the rows of 2017 and 2019 in `data/gravity_extract.csv` (committed). Only runs with `--download`. |
| 2 | `code/02_clean.ipynb` | Cleans the extract and saves `data/gravity_clean.csv`, the table every model uses, and `results/cleaning_log.csv`. |
| 3 | `code/03_development.ipynb` | Fits the models on training samples of every size in the development year (2017) and saves `results/dev_runs.csv` and `results/dev_components.csv`. |
| 3b | `code/03b_checkpoint.ipynb` | The checkpoint report on part 1 of stage 3, in `results/checkpoint/`. |
| 4 | `code/04_size_rule.ipynb` | Tunes the ensembles, builds the size rule and finds the training size where the models cross, from the files of stage 3. |
| 5 | `code/05_analysis.ipynb` | Tests the cutoff and the size rule against cross-validation in the analysis year (2019), at the settings fixed in stage 4. |
| 6 | `code/06_checks.ipynb` | Repeats the development-year experiment with added noise, with another test split and with twice the trees. |
| 7 | `code/07_figures.ipynb` | Makes every figure and table shown in the paper, in `figures/`. |

`code/03b_checkpoint.ipynb` is not in the order of a full run. It was run once, between part 1 and the full run of stage 3, to review the results of the training sizes up to 1,000 under rules fixed in advance (section 6 of [`docs/build_plan.md`](docs/build_plan.md)). It was run on 20 October 2026, on the files of part 1 copied into `results/checkpoint/inputs/`. Quick mode runs it after stage 3, so that every push tests it.

`code/helpers.py` holds what more than one notebook uses. [`docs/build_plan.md`](docs/build_plan.md) describes every stage in detail.

## Time and output

- Running time: TODO, measure in the Nuvolos workspace.
- Resources beyond the Nuvolos workspace: none. Stage 1 downloads 207 MB from CEPII and needs about 1.5 GB of free disk space; the other stages start from the committed extract.
- Output: the result tables in `results/` (listed in [`results/README.md`](results/README.md)) and the figures and tables of the paper in `figures/` (listed in [`figures/README.md`](figures/README.md)).

## Data

Source, licence and cleaning steps are in [`data/README.md`](data/README.md).

## Paper and recording

- Paper: TODO, `paper.pdf` in this folder.
- Recording: TODO, link.
