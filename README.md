# Semester project: Advanced Data Analysis, HEC Lausanne, autumn 2026

<!-- Outline drafted with Claude (Anthropic). Asked for: a README skeleton in the order given in the course document "Starting and Shipping Your Project". -->

## What this project does

TODO, after the proposal is signed off: two sentences on the question and on what the code produces.

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

| Order | File | What it does |
|---|---|---|
| 1 | `code/01_get_data.ipynb` | Downloads the CEPII Gravity file into `data/raw/` (not committed), checks it, and saves the rows of 2017 and 2019 in `data/gravity_extract.csv` (committed). Only runs with `python code/run_all.py --download`. |
| 2 | `code/02_clean.ipynb` | Cleans `data/gravity_extract.csv` and saves the table the models use. |
| 3 | `code/03_model.ipynb` | Fits the models and saves the results. |
| 4 | `code/04_figures.ipynb` | Makes every figure and table shown in the paper. |

`code/helpers.py` holds what more than one notebook uses.

## Time and output

- Running time: TODO, measure in the Nuvolos workspace.
- Resources beyond the Nuvolos workspace: none. Stage 1 downloads 207 MB from CEPII and needs about 1.5 GB of free disk space; the other stages start from the committed extract.
- Output: the figures and tables of the paper, saved in `figures/`.

## Data

Source, licence and cleaning steps are in [`data/README.md`](data/README.md).

## Paper and recording

- Paper: TODO, `paper.pdf` in this folder.
- Recording: TODO, link.
