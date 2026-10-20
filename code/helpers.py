# Drafted with Claude (Anthropic). Asked for: folder locations that work on
# any computer, so that no notebook needs an absolute path, and one random
# seed shared by every notebook. Later asked for: the folder of the raw
# download and the two years of the project. Later asked for (18 October
# 2026, section 5 of docs/build_plan.md): one thread per model, a quick mode
# for testing the code, the parameters of every stage, and random generators
# built from the one seed. Later asked for (20 October 2026, decisions 15 and 17
# of docs/build_plan.md): the switch and penalty grid of the ridge line, and the
# number of keys in the seed of each random generator.
"""Code used by more than one notebook (Requirement 4.9).

Every notebook imports this module in its first code cell, before pandas,
numpy or scikit-learn:

    from helpers import DATA, RESULTS, SEED
"""
import os
import sys

# One thread per model. The long stages spread the training draws over
# workers (ADA_JOBS), so a model that also used every core would compete with
# the other workers, and a worker started after the main process has fitted a
# boosted model can hang (section 7 of the build plan). These variables only
# take effect if they are set before numpy is first imported, so this module
# has to be imported first.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
if "numpy" in sys.modules:
    raise ImportError(
        "numpy was imported before helpers, so the one-thread settings cannot "
        "take effect. Put 'from helpers import ...' at the top of the first "
        "code cell, above pandas, numpy and scikit-learn, and restart the kernel."
    )

from pathlib import Path

import numpy as np

# The repository folder, wherever it has been cloned.
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
# The CEPII download: too large for GitHub, so .gitignore leaves it out.
RAW = DATA / "raw"

# Quick mode only tests that the code runs: few sizes, few draws, small
# models. Its results and figures go into folders that git ignores, and
# nothing it writes is evidence. Stage 2, which quick mode does not change,
# still writes data/gravity_clean.csv. Start it with ADA_QUICK=1 python
# code/run_all.py.
QUICK = os.environ.get("ADA_QUICK") == "1"
RESULTS = ROOT / ("results_quick" if QUICK else "results")
FIGURES = ROOT / ("figures_quick" if QUICK else "figures")
if QUICK:
    print("Quick mode (ADA_QUICK=1): this run only tests that the code runs. "
          "Its results and figures go to results_quick/ and figures_quick/ and are not evidence; "
          "stage 2 still writes data/gravity_clean.csv.")

# One seed for every random step, so the numbers in the paper can be reproduced.
SEED = 2026

# The development year (for exploring and tuning) and the analysis year (for
# confirming), chosen by Yu on 6 October 2026.
DEV_YEAR = 2017
ANALYSIS_YEAR = 2019

# The table every model uses: the flow to predict, in thousand current USD and
# in levels, and the eight predictors, the same for every model (section 1).
TARGET = "tradeflow_baci"
INDICATORS = ["contig", "comlang_off", "col_dep_ever", "comcol", "fta_wto"]
PREDICTORS = ["ln_gdp_o", "ln_gdp_d", "ln_dist"] + INDICATORS

# Parameters of the stages (section 5 of the build plan). Where a line reads
# `A if QUICK else B`, A is the quick-mode value and B the full-run value.
TEST_SHARE = 0.25  # share of unordered pairs in the test set
SIZES = (50, 200, 1000) if QUICK else (50, 100, 200, 500, 1000, 2000, 5000)
CHECKPOINT_MAX = 200 if QUICK else 1000  # largest size of part 1 (decision 11)
DRAWS = 4 if QUICK else 50
RF_TREES = 20 if QUICK else 100
RF_MAX_FEATURES = (1.0, 3)  # 1.0 means all eight predictors, 3 means three
RF_MIN_LEAF = (1, 5, 20)
GBM_MAX_TREES = 100 if QUICK else 500
GBM_TREE_COUNTS = (25, 50, 100) if QUICK else (25, 50, 100, 200, 300, 400, 500)
GBM_DEPTHS = (1, 2, 3, 5)
GBM_LEARNING_RATE = 0.1
GBM_MIN_LEAF = 20  # decision 9
CV_FOLDS = 5
CV_DRAWS = 2 if QUICK else 50  # decision 4
BOOTSTRAP = 50 if QUICK else 1000
MIN_FLOW = None  # decision 3: no floor, every recorded flow is kept
NOISE_VAR = 1.0  # decision 5
# Decision 15: a ridge-penalised ppml as a third reference line. It runs only if
# Yu adds it at the checkpoint (section 6 of the build plan, Ridge line); its
# penalty grid is fixed here, before part 1.
RIDGE_LINE = False
RIDGE_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1, 10)
N_JOBS = int(os.environ.get("ADA_JOBS", "1"))  # decision 6

# Part 1 of stage 3 (ADA_PART=1) runs the sizes up to CHECKPOINT_MAX, before
# the checkpoint. Unset, stage 3 runs every size. Only 03_development.ipynb
# reads it: stages 2 and 4 to 7 call refuse_part(). Stage 1 does not (section 5
# item 2 of the plan leaves it unchanged), and run_all.py never passes ADA_PART
# to it.
_part = os.environ.get("ADA_PART")
if _part not in (None, "1"):
    raise ValueError(f"ADA_PART is {_part!r}: it can only be 1, or unset to run every size.")
PART = None if _part is None else int(_part)


def refuse_part():
    """Stop a notebook other than stage 3 when ADA_PART is set."""
    if PART is not None:
        raise RuntimeError(
            "ADA_PART is set, but only 03_development.ipynb runs in parts. "
            "Run part 1 with: python code/run_all.py --part 1"
        )


# Each random step has its own purpose code, so that, for example, the draws
# of training samples and the seeds of the models never share numbers.
PURPOSES = {"split": 1, "draw": 2, "model": 3, "noise": 4, "cv": 5, "bootstrap": 6}


def rng(purpose, *keys):
    """A random generator for one step, built from SEED, the purpose and the keys.

    The keys are integers from 0 to 2**32 - 1 that name the step, such as the
    year, the split, the training size and the draw. The same purpose and keys
    always give the same numbers, so a result does not depend on the order of
    the loops or on the number of workers.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose {purpose!r}: use one of {sorted(PURPOSES)}.")
    for key in keys:
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 0 <= key < 2**32:
            raise ValueError(f"Keys must be integers from 0 to 2**32 - 1, got {key!r}.")
    # The number of keys is part of the seed (decision 17). numpy pads a seed
    # shorter than four numbers with zeros, so without it rng("split", 2017)
    # and rng("split", 2017, 0) would give the same numbers. A key below 2**32
    # is one number of the seed, so two different lists of keys always give
    # different seeds.
    return np.random.default_rng([SEED, PURPOSES[purpose], len(keys), *[int(key) for key in keys]])


def seed_int(purpose, *keys):
    """One integer drawn from rng(purpose, *keys), for scikit-learn's random_state."""
    return int(rng(purpose, *keys).integers(0, 2**31 - 1))
