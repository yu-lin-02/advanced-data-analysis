# Drafted with Claude (Anthropic). Asked for: folder locations that work on
# any computer, so that no notebook needs an absolute path, and one random
# seed shared by every notebook. Later asked for: the folder of the raw
# download and the two years of the project. Later asked for (18 October
# 2026, section 5 of docs/build_plan.md): one thread per model, a quick mode
# for testing the code, the parameters of every stage, and random generators
# built from the one seed. Later asked for (20 October 2026, decisions 15 and 17
# of docs/build_plan.md): the switch and penalty grid of the ridge line, and the
# number of keys in the seed of each random generator. Later asked for (20
# October 2026, step 2 of section 10 of docs/build_plan.md): the functions of
# stage 3 (the test split, one training draw with its 13 fits, the rows it
# writes, the split of the error into variance and remainder) and those of the
# checkpoint report that stages 4, 5 and 7 will reuse. Later asked for (21
# October 2026, decision 19 of docs/build_plan.md, "Apply"; prepared on 20
# October before Yu's answer): the boosting grid extended as the rule Grid
# edges of the checkpoint called for, with the report of grid edges read on
# the grid its inputs were made on. Later asked for (21 October 2026, decision
# 20 of docs/build_plan.md, "Add it"): the ridge line, a ridge-penalised ppml
# whose penalty is chosen by cross-validation inside each training sample,
# fitted on every draw from the full run of stage 3 on, and never in part 1.
# Later asked for (22 October 2026, step 6 of section 10 of docs/build_plan.md):
# the functions of stage 4 (the test of the rule Stability after the full run
# of stage 3, the table of the size rule, the gaps with the win counts of
# ppml, and the bootstrap of the cutoff). Later asked for (22 October 2026,
# step 7 of section 10 of docs/build_plan.md, after the freeze of decision 21):
# the functions of stage 5 (the fits at the frozen settings, cross-validated
# selection on the folds of the ridge line, the summary with the test of H2,
# the cutoff of H1 and of decision 22 with its bootstrap, and the selection
# table), with the folds of the ridge line moved into their own function and
# forests that can be fitted without out-of-bag predictions. Later asked for (23
# October 2026, step 7 of section 10 of docs/build_plan.md, after Yu's run of
# stage 5): the functions of stage 6 (noise added to the training flows of a
# draw, one whole experiment of stage 3 as a function that stage 3 and stage 6
# share, the steps of stage 4 on the files of one experiment, and the check
# with twice the trees), with the scores of one configuration on the test rows
# in their own function, which that check reuses.
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

import concurrent.futures
import multiprocessing
import time
import warnings
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (d2_tweedie_score, mean_absolute_error, mean_poisson_deviance,
                             root_mean_squared_error)
from sklearn.tree import DecisionTreeRegressor

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
# Decision 19: the checkpoint's rule Grid edges added depth 7 (in quick mode
# too) and, in the full run, 750 and 1,000 trees with GBM_MAX_TREES 1,000. The
# first run of part 1 had depths 1, 2, 3 and 5 and at most 500 trees.
GBM_MAX_TREES = 100 if QUICK else 1000
GBM_TREE_COUNTS = (25, 50, 100) if QUICK else (25, 50, 100, 200, 300, 400, 500, 750, 1000)
GBM_DEPTHS = (1, 2, 3, 5, 7)
GBM_LEARNING_RATE = 0.1
GBM_MIN_LEAF = 20  # decision 9
CV_FOLDS = 5
CV_DRAWS = 2 if QUICK else 50  # decision 4
BOOTSTRAP = 50 if QUICK else 1000
MIN_FLOW = None  # decision 3: no floor, every recorded flow is kept
NOISE_VAR = 1.0  # decision 5
# Decisions 15, 18 and 20: a ridge-penalised ppml as a third reference line,
# which Yu added at the checkpoint (section 6 of the build plan, Ridge line). Its
# penalty grid was fixed before part 1. Both runs of part 1 were made while
# this switch was still False, and FIT_RIDGE below keeps part 1 without the
# ridge line, so no file of part 1 has a ridge row.
RIDGE_LINE = True
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
# The ridge line is fitted only outside part 1, so that no file of part 1 ever
# has a ridge row (decision 20), even if part 1 is run again under the
# checkpoint's rule Bugs.
FIT_RIDGE = RIDGE_LINE and PART is None

# The checkpoint report (section 6 of the plan) reads the two files of part 1
# from checkpoint/inputs/ and writes its report next to them. If the checkpoint
# calls for a rerun of part 1, ADA_CHECKPOINT=rerun points it at
# checkpoint/rerun/ instead.
_checkpoint = os.environ.get("ADA_CHECKPOINT")
if _checkpoint not in (None, "rerun"):
    raise ValueError(f"ADA_CHECKPOINT is {_checkpoint!r}: it can only be rerun, or unset.")
CHECKPOINT = RESULTS / "checkpoint"
if _checkpoint == "rerun":
    CHECKPOINT = CHECKPOINT / "rerun"


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


# ---------------------------------------------------------------------------
# Stage 3: the test split and one training draw (sections 6 and 7 of the plan)
# ---------------------------------------------------------------------------


def rf_features(max_features):
    """The number of predictors a forest considers at each split.

    scikit-learn reads a decimal such as 1.0 as a share of the predictors (1.0
    is all eight) and a whole number such as 3 as a count, so the files record
    the count, which reads the same way in both cases.
    """
    if isinstance(max_features, float):
        return round(max_features * len(PREDICTORS))
    return max_features


def model_configurations():
    """The configurations scored on every draw: one model at one setting each (section 1).

    Built from the parameters above, so that the counts follow the grid if the
    checkpoint extends it. The order is the order of the rows of each draw.
    """
    configurations = [{"model": "ppml", "config": "ppml"}]
    for max_features in RF_MAX_FEATURES:
        for min_leaf in RF_MIN_LEAF:
            features = rf_features(max_features)
            configurations.append({"model": "rf", "config": f"rf_features{features}_leaf{min_leaf}",
                                   "rf_features": features, "rf_min_leaf": min_leaf})
    for depth in GBM_DEPTHS:
        for trees in GBM_TREE_COUNTS:
            configurations.append({"model": "gbm", "config": f"gbm_depth{depth}_trees{trees}",
                                   "gbm_depth": depth, "gbm_trees": trees})
    configurations.append({"model": "mean", "config": "mean"})
    configurations.append({"model": "tree", "config": "tree"})
    if FIT_RIDGE:
        configurations.append({"model": "ridge", "config": "ridge"})
    return configurations


CONFIGURATIONS = model_configurations()
# The three groups of test rows whose deviance stage 3 records separately.
GROUP_NAMES = ("top1", "next9", "rest")
# The columns of dev_runs.csv, in order (stage 3 step 5). lbfgs_fallback,
# train_total_ratio and the coef_ and ones_ columns are filled on the ppml rows
# only (decision 14). ridge_alpha, the penalty that cross-validation chose on
# the draw, is filled on the ridge rows only; it comes last, so the columns
# before it are in the same places as in the files of part 1.
RUN_COLUMNS = (
    ["variant", "year", "split", "size", "draw", "model", "config",
     "rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees",
     "test_deviance", "d2", "mae_rel", "rmse_rel", "train_deviance"]
    + [f"deviance_{name}" for name in GROUP_NAMES]
    + ["max_pred_ratio", "seconds", "constant_columns", "lbfgs_fallback", "train_total_ratio", "coef_intercept"]
    + [f"coef_{name}" for name in PREDICTORS]
    + [f"ones_{name}" for name in INDICATORS]
    + ["ridge_alpha"]
)
# Columns that hold whole numbers but are empty on some rows. pandas would
# write them as 8.0; the type Int64 writes 8, and an empty field where missing.
WHOLE_NUMBER_COLUMNS = (["rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees", "constant_columns", "lbfgs_fallback"]
                        + [f"ones_{name}" for name in INDICATORS])


def split_pairs(table, year, split_id):
    """Split one year into a training pool and a test set by unordered pair (section 7).

    The distinct pair_id values are sorted, permuted with rng("split", year,
    split_id), and the first quarter of them (TEST_SHARE, rounded down) goes to
    the test set with all its rows. Both directions of a pair therefore land on
    the same side. Rows keep the order they have in the table.
    """
    rows = table[table["year"] == year]
    pairs = np.sort(rows["pair_id"].unique())
    permuted = rng("split", year, split_id).permutation(pairs)
    test_pairs = permuted[: int(TEST_SHARE * len(pairs))]
    in_test = rows["pair_id"].isin(test_pairs)
    pool = rows[~in_test].reset_index(drop=True)
    test = rows[in_test].reset_index(drop=True)
    assert not set(pool["pair_id"]) & set(test["pair_id"]), "A pair_id is on both sides of the split."
    return pool, test


def test_groups(y_test):
    """The group of each test row: 0 for the largest 1% of test flows, 1 for the next 9%, 2 for the rest.

    Fixed once from the test flows of the split, so every model and every draw
    is scored on the same rows. Rows are ranked by flow, largest first. The
    counts of the largest 1% and of the largest 10% are rounded down, and
    group 1 is the largest 10% without group 0 (so it can be one row more than
    9% rounded down). Equal flows keep their order in the test set.
    """
    order = np.argsort(-y_test, kind="stable")
    top = int(0.01 * len(y_test))
    top_and_next = int(0.10 * len(y_test))
    groups = np.full(len(y_test), 2)
    groups[order[:top]] = 0
    groups[order[top:top_and_next]] = 1
    return groups


def unit_deviance(y, predicted):
    """The Poisson deviance of each row, d(y, m) = 2 * (y * ln(y / m) - y + m) (section 1).

    Every recorded flow is positive, so y * ln(y / m) needs no special case for
    y = 0. Its mean equals sklearn's mean_poisson_deviance.
    """
    return 2 * (y * np.log(y / predicted) - y + predicted)


def draw_sample(pool_rows, year, split_id, n, draw):
    """Positions in the pool of one training sample: n distinct rows, drawn without replacement."""
    positions = rng("draw", year, split_id, n, draw).choice(pool_rows, size=n, replace=False)
    assert len(np.unique(positions)) == n, "A training sample does not have n distinct pool rows."
    return positions


def noise_factor(rows, year, split_id, n, draw):
    """The factors that multiply the training flows of one draw in the check noise of stage 6 (decision 5).

    Each factor is exp(s * z - s**2 / 2), with z standard normal from
    rng("noise", year, split_id, n, draw) and s**2 = ln(1 + NOISE_VAR), so it
    has mean 1 and variance NOISE_VAR: the expected flow stays the same and
    its variance grows, as in case 3 of Santos Silva and Tenreyro (2006, p. 647).
    One independent factor per training row, drawn fresh for every draw.
    """
    s = np.sqrt(np.log(1 + NOISE_VAR))
    z = rng("noise", year, split_id, n, draw).standard_normal(rows)
    return np.exp(s * z - s**2 / 2)


def make_ppml():
    """The gravity equation estimated by PPML. alpha=0: the default alpha=1 would add a ridge penalty."""
    return PoissonRegressor(alpha=0, solver="newton-cholesky")


def make_ridge(alpha):
    """ppml with a ridge penalty of strength alpha on the coefficients (the ridge line). The intercept is not penalised."""
    return PoissonRegressor(alpha=alpha, solver="newton-cholesky")


def make_rf(max_features, min_leaf, seed, oob=True, trees=RF_TREES):
    """A random forest with the Poisson loss, fitted on one thread.

    oob_score=True keeps the out-of-bag predictions for the training deviance
    of stage 3. It leaves the predictions unchanged, so stage 5, which records
    no training deviance, fits its forests with oob=False and times the fit alone.
    trees is RF_TREES except in the check trees2x of stage 6, which doubles it.
    """
    return RandomForestRegressor(n_estimators=trees, criterion="poisson", max_features=max_features,
                                 min_samples_leaf=min_leaf, oob_score=oob, n_jobs=1, random_state=seed)


def make_gbm(depth, seed, trees=GBM_MAX_TREES):
    """Gradient-boosted trees with the Poisson loss. Every tree is kept: no early stopping."""
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=GBM_LEARNING_RATE, max_iter=trees,
                                         max_depth=depth, min_samples_leaf=GBM_MIN_LEAF, max_leaf_nodes=None,
                                         early_stopping=False, random_state=seed)


def make_tree(seed):
    """One decision tree with the Poisson loss, grown to full depth (a reference line)."""
    return DecisionTreeRegressor(criterion="poisson", random_state=seed)


def fit_ppml(X_train, y_train, X_test):
    """Fit ppml on one training sample.

    An indicator that is constant in the sample is dropped from this one fit
    (its coefficient is in effect zero), because PoissonRegressor(alpha=0)
    needs a design matrix of full rank (stage 3). If the other columns are
    collinear, scikit-learn warns and falls back from Newton-Cholesky to
    L-BFGS: that is recorded, not repaired. What is recorded is whether
    scikit-learn warned that it fell back, as the plan words it.
    """
    indicator_columns = [PREDICTORS.index(name) for name in INDICATORS]
    constant = [j for j in indicator_columns if X_train[:, j].min() == X_train[:, j].max()]
    kept = [j for j in range(len(PREDICTORS)) if j not in constant]
    model = make_ppml()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        model.fit(X_train[:, kept], y_train)
        seconds = time.perf_counter() - start
    fell_back = False
    for warning in caught:
        if "lbfgs" in str(warning.message):
            fell_back = True
        else:
            # Any other warning is shown as usual.
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    coefficients = {name: np.nan for name in PREDICTORS}
    for j, value in zip(kept, model.coef_):
        coefficients[PREDICTORS[j]] = value
    train_predicted = model.predict(X_train[:, kept])
    return {
        "test_predicted": model.predict(X_test[:, kept]),
        "train_predicted": train_predicted,
        "intercept": model.intercept_,
        "coefficients": coefficients,
        "ones": {name: int(X_train[:, PREDICTORS.index(name)].sum()) for name in INDICATORS},
        "constant_columns": len(constant),
        "fell_back": fell_back,
        "seconds": seconds,
        # Section 8: unpenalised PPML predicts the observed training total, unless the fit fell back.
        "total_ratio": train_predicted.sum() / y_train.sum(),
    }


def fit_ridge_once(alpha, X_fit, y_fit, X_other):
    """One fit of the ridge line on the rows X_fit, y_fit (section 6, Ridge line).

    Each predictor is standardised with its mean and standard deviation over
    the rows of this fit (ddof 0, as StandardScaler computes it), so that the
    penalty weighs every coefficient on the same scale. A column with one
    value on those rows cannot be standardised and is dropped from this fit,
    as ppml drops a constant indicator. The flow is divided by its mean over
    the same rows, so that alpha means the same at every size and in both
    years (decision 18), and the predictions are multiplied back into thousand
    USD. Returns the predictions for the rows X_other and for the fit rows,
    and the seconds of the fit call.
    """
    kept = [j for j in range(X_fit.shape[1]) if X_fit[:, j].min() != X_fit[:, j].max()]
    centre = X_fit[:, kept].mean(axis=0)
    spread = X_fit[:, kept].std(axis=0)
    # The other rows are standardised with the fit rows' mean and spread, as the fit saw them.
    Z_fit = (X_fit[:, kept] - centre) / spread
    Z_other = (X_other[:, kept] - centre) / spread
    level = y_fit.mean()
    model = make_ridge(alpha)
    start = time.perf_counter()
    model.fit(Z_fit, y_fit / level)
    seconds = time.perf_counter() - start
    return {
        "other_predicted": level * model.predict(Z_other),
        "fit_predicted": level * model.predict(Z_fit),
        "seconds": seconds,
    }


def cv_folds(pair_train, year, split_id, n, draw):
    """The fold, from 0 to CV_FOLDS - 1, of each row of one training sample.

    The folds keep both directions of a pair together, as the test split does:
    the distinct pair_id values of the sample, sorted, are permuted with
    rng("cv", year, split_id, n, draw) and dealt to the folds in turn (section
    6, Ridge line). The ridge line and cv_select (stage 5) use the same folds
    on the same sample.
    """
    # np.unique returns the distinct values sorted.
    permuted = rng("cv", year, split_id, n, draw).permutation(np.unique(pair_train))
    # Dealt in turn: the pair in place i goes to fold i mod 5 (0, 1, 2, 3, 4, 0, 1, ...).
    fold_of_pair = {pair: i % CV_FOLDS for i, pair in enumerate(permuted)}
    return np.array([fold_of_pair[pair] for pair in pair_train])


def fit_ridge(X_train, y_train, pair_train, X_test, year, split_id, n, draw):
    """Fit the ridge line on one training sample (section 6, Ridge line; decisions 15, 18 and 20).

    5-fold cross-validation inside the sample picks alpha from RIDGE_ALPHAS:
    the value with the lowest mean, over the five folds, of the mean Poisson
    deviance on the validation fold, in thousand USD. A tie goes to the larger
    value, the simpler model. The folds keep both directions of a pair
    together (cv_folds). The test rows play no part in choosing the
    penalty. The ridge line is then refitted on the whole sample at the chosen
    value, and only this refit predicts the test rows: 31 fits in all, whose
    seconds are summed.
    """
    folds = cv_folds(pair_train, year, split_id, n, draw)
    seconds = 0.0
    cv_deviance = {}
    for alpha in RIDGE_ALPHAS:
        fold_deviances = []
        for fold in range(CV_FOLDS):
            validation = folds == fold
            fit = fit_ridge_once(alpha, X_train[~validation], y_train[~validation], X_train[validation])
            seconds += fit["seconds"]
            fold_deviances.append(mean_poisson_deviance(y_train[validation], fit["other_predicted"]))
        cv_deviance[alpha] = np.mean(fold_deviances)
    # From the largest value down, a value replaces the one chosen so far only
    # if its deviance is strictly lower, so a tie stays with the larger value.
    chosen = max(RIDGE_ALPHAS)
    for alpha in sorted(RIDGE_ALPHAS, reverse=True):
        if cv_deviance[alpha] < cv_deviance[chosen]:
            chosen = alpha
    refit = fit_ridge_once(chosen, X_train, y_train, X_test)
    seconds += refit["seconds"]
    return {"test_predicted": refit["other_predicted"], "train_predicted": refit["fit_predicted"],
            "alpha": chosen, "seconds": seconds}


def run_draw(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draw, noise=False):
    """Fit every model on one training sample and predict the test rows for every configuration (section 7).

    The sample is drawn with rng("draw", year, split_id, n, draw) and every
    model gets the same random_state, seed_int("model", year, split_id, n,
    draw), so a draw gives the same numbers whichever worker runs it. The
    ridge line (when FIT_RIDGE is True) draws its folds from rng("cv", ...)
    and the pair_id values of the sample (pair_pool holds those of the pool),
    so it changes no random number of the other models. Returns, in the order
    of CONFIGURATIONS, the test predictions, the mean Poisson deviance on the
    training sample (out-of-bag for rf, none for tree) and the seconds of the
    fit each configuration came from (the fit call alone, as stage 3 step 5
    defines them: predicting the test rows and scoring are not timed, but the
    fit of a forest includes the out-of-bag predictions that oob_score=True
    makes scikit-learn compute, and the ridge line sums its 31 fits); and what
    stage 3 records about the sample, the ppml fit and the ridge line.

    With noise=True (the check noise of stage 6), the training flows of the
    sample are multiplied by noise_factor before any model sees them: the same
    sample and the same seeds as without noise, and the test flows untouched.
    Everything computed on the training sample (the training deviance, the
    largest training flow, the training total of ppml, the folds of the ridge
    line) then uses the noisy flows.
    """
    assert max(GBM_TREE_COUNTS) <= GBM_MAX_TREES, "A tree count of GBM_TREE_COUNTS is above GBM_MAX_TREES."
    positions = draw_sample(len(y_pool), year, split_id, n, draw)
    X_train, y_train = X_pool[positions], y_pool[positions]
    if noise:
        y_train = y_train * noise_factor(n, year, split_id, n, draw)
    seed = seed_int("model", year, split_id, n, draw)
    test_predicted, train_deviance, seconds = {}, {}, {}

    ppml = fit_ppml(X_train, y_train, X_test)
    seconds["ppml"] = ppml["seconds"]
    test_predicted["ppml"] = ppml["test_predicted"]
    train_deviance["ppml"] = mean_poisson_deviance(y_train, ppml["train_predicted"])

    for max_features in RF_MAX_FEATURES:
        for min_leaf in RF_MIN_LEAF:
            config = f"rf_features{rf_features(max_features)}_leaf{min_leaf}"
            start = time.perf_counter()
            with warnings.catch_warnings():
                # With few trees a training row can be in every tree's sample, so it has
                # no out-of-bag prediction; such rows are left out of the training deviance below.
                warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
                forest = make_rf(max_features, min_leaf, seed).fit(X_train, y_train)
            seconds[config] = time.perf_counter() - start
            test_predicted[config] = forest.predict(X_test)
            # scikit-learn leaves the out-of-bag prediction at 0 for a row that no tree left out.
            has_oob = forest.oob_prediction_ > 0
            train_deviance[config] = mean_poisson_deviance(y_train[has_oob], forest.oob_prediction_[has_oob])

    for depth in GBM_DEPTHS:
        start = time.perf_counter()
        boosted = make_gbm(depth, seed).fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        # One fit with GBM_MAX_TREES trees gives the predictions after every
        # count in GBM_TREE_COUNTS (section 7, staged prediction).
        for trees, predicted in enumerate(boosted.staged_predict(X_test), start=1):
            if trees in GBM_TREE_COUNTS:
                test_predicted[f"gbm_depth{depth}_trees{trees}"] = predicted
        for trees, predicted in enumerate(boosted.staged_predict(X_train), start=1):
            if trees in GBM_TREE_COUNTS:
                train_deviance[f"gbm_depth{depth}_trees{trees}"] = mean_poisson_deviance(y_train, predicted)
        for trees in GBM_TREE_COUNTS:
            seconds[f"gbm_depth{depth}_trees{trees}"] = elapsed

    # The "fit" of mean is the mean of the training flows.
    start = time.perf_counter()
    mean_flow = y_train.mean()
    seconds["mean"] = time.perf_counter() - start
    test_predicted["mean"] = np.full(len(y_test), mean_flow)
    train_deviance["mean"] = mean_poisson_deviance(y_train, np.full(len(y_train), mean_flow))

    start = time.perf_counter()
    tree = make_tree(seed).fit(X_train, y_train)
    seconds["tree"] = time.perf_counter() - start
    test_predicted["tree"] = tree.predict(X_test)
    # A tree grown to full depth fits its own sample exactly, so its training deviance is not recorded.
    train_deviance["tree"] = np.nan

    ridge = None
    if FIT_RIDGE:
        ridge = fit_ridge(X_train, y_train, pair_pool[positions], X_test, year, split_id, n, draw)
        seconds["ridge"] = ridge["seconds"]
        test_predicted["ridge"] = ridge["test_predicted"]
        train_deviance["ridge"] = mean_poisson_deviance(y_train, ridge["train_predicted"])

    order = [configuration["config"] for configuration in CONFIGURATIONS]
    return {
        "test_predicted": np.array([test_predicted[config] for config in order]),
        "train_deviance": np.array([train_deviance[config] for config in order]),
        "seconds": np.array([seconds[config] for config in order]),
        "largest_train_flow": y_train.max(),
        "constant_columns": ppml["constant_columns"],
        "ppml": ppml,
        "ridge": ridge,
    }


def run_draws(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draws, noise=False):
    """Run the draws 0 to draws - 1 of one size and yield their results in the order of the draws.

    With N_JOBS = 1 the draws run one after the other in this process. With
    more, they are spread over N_JOBS worker processes, started with
    "forkserver" (section 7): a worker forked from a process that has already
    fitted a boosted model can hang. Each draw is keyed by its own numbers, and
    the results come back in the order of the draws, so nothing depends on the
    number of workers. noise is passed on to run_draw.
    """
    if N_JOBS == 1:
        for draw in range(draws):
            yield run_draw(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draw, noise)
        return
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS, mp_context=context) as executor:
        yield from executor.map(run_draw, repeat(X_pool), repeat(y_pool), repeat(pair_pool), repeat(X_test),
                                repeat(y_test), repeat(year), repeat(split_id), repeat(n), range(draws),
                                repeat(noise))


def check_predictions(predicted, keys):
    """Stop unless every prediction is positive and finite (section 8): stage 3 takes their logs."""
    good = np.isfinite(predicted).all(axis=1) & (predicted > 0).all(axis=1)
    bad = [configuration["config"] for configuration, ok in zip(CONFIGURATIONS, good) if not ok]
    assert not bad, f"Predictions that are not positive and finite at {keys}: {bad}. Stop and report."


def test_scores(y_test, predicted, groups, where):
    """The scores of one configuration on the test rows (stage 3 step 5).

    The mean Poisson deviance, D squared, MAE and RMSE divided by the mean test
    flow, and the deviance of each group of test rows (test_groups), summed
    over its rows and divided by all the test rows, so that the three add up to
    the test deviance. where names the draw and configuration in the message
    if they do not.
    """
    mean_test_flow = y_test.mean()
    scores = {
        "test_deviance": mean_poisson_deviance(y_test, predicted),
        "d2": d2_tweedie_score(y_test, predicted, power=1),
        "mae_rel": mean_absolute_error(y_test, predicted) / mean_test_flow,
        "rmse_rel": root_mean_squared_error(y_test, predicted) / mean_test_flow,
    }
    unit = unit_deviance(y_test, predicted)
    for group, name in enumerate(GROUP_NAMES):
        scores[f"deviance_{name}"] = unit[groups == group].sum() / len(y_test)
    groups_total = sum(scores[f"deviance_{name}"] for name in GROUP_NAMES)
    assert np.isclose(groups_total, scores["test_deviance"], rtol=1e-8, atol=0), \
        f"The three groups do not add up to the test deviance at {where}."
    return scores


def draw_rows(result, y_test, groups, keys):
    """The rows of one draw for dev_runs.csv, one per configuration (stage 3 step 5).

    keys holds the columns that name the draw: variant, year, split, size and draw.
    """
    rows = []
    for i, configuration in enumerate(CONFIGURATIONS):
        predicted = result["test_predicted"][i]
        row = dict(keys)
        row.update(configuration)
        row.update(test_scores(y_test, predicted, groups, f"{keys}, {configuration['config']}"))
        row["train_deviance"] = result["train_deviance"][i]
        row["max_pred_ratio"] = predicted.max() / result["largest_train_flow"]
        row["seconds"] = result["seconds"][i]
        row["constant_columns"] = result["constant_columns"]
        if configuration["model"] == "ppml":
            ppml = result["ppml"]
            row["lbfgs_fallback"] = int(ppml["fell_back"])
            row["train_total_ratio"] = ppml["total_ratio"]
            row["coef_intercept"] = ppml["intercept"]
            for name in PREDICTORS:
                row[f"coef_{name}"] = ppml["coefficients"][name]
            for name in INDICATORS:
                row[f"ones_{name}"] = ppml["ones"][name]
        if configuration["model"] == "ridge":
            row["ridge_alpha"] = result["ridge"]["alpha"]
        rows.append(row)
    return rows


def runs_table(rows):
    """The rows of draw_rows as a table, with the columns of dev_runs.csv in order."""
    table = pd.DataFrame(rows).reindex(columns=RUN_COLUMNS)
    for column in WHOLE_NUMBER_COLUMNS:
        table[column] = table[column].astype("Int64")
    return table


# The columns of dev_components.csv, in order (stage 3 step 7).
COMPONENT_COLUMNS = ["variant", "year", "split", "size", "model", "config",
                     "rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees", "draws", "pool_rows", "error",
                     "variance_raw", "remainder_raw", "variance_r", "remainder_r", "variance_rn", "remainder_rn"]


def components_table(rows):
    """The rows of component_rows as a table, with the columns of dev_components.csv in order."""
    table = pd.DataFrame(rows).reindex(columns=COMPONENT_COLUMNS)
    for column in ["rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees"]:
        table[column] = table[column].astype("Int64")
    return table


def split_error(sum_predicted, sum_log_predicted, y_test, deviance_per_draw, n, pool_rows):
    """Split the mean test deviance of one configuration into variance and remainder (section 7).

    With R draws, for each test row the arithmetic mean prediction is
    am = sum_predicted / R and the geometric mean is gm = exp(sum_log_predicted / R).
    variance_raw is the mean over test rows of 2 * (am - gm), the average
    deviance between each draw's prediction and gm. remainder_raw is the mean
    over test rows of d(y, gm): noise plus bias. Their sum is the mean test
    deviance over draws. variance_r corrects for the number of draws (R / (R - 1)),
    and variance_rn also for the overlap of samples drawn from one pool
    (divided by 1 - n / pool_rows). Each remainder is the error minus the
    matching variance.
    """
    draws = len(deviance_per_draw)
    arithmetic_mean = sum_predicted / draws
    geometric_mean = np.exp(sum_log_predicted / draws)
    error = float(np.mean(deviance_per_draw))
    variance_raw = float(np.mean(2 * (arithmetic_mean - geometric_mean)))
    remainder_raw = float(np.mean(unit_deviance(y_test, geometric_mean)))
    assert np.isclose(error, remainder_raw + variance_raw, rtol=1e-8, atol=0), \
        f"error {error} is not remainder_raw + variance_raw = {remainder_raw + variance_raw} (section 8)."
    variance_r = variance_raw * draws / (draws - 1)
    variance_rn = variance_r / (1 - n / pool_rows)
    return {
        "draws": draws, "pool_rows": pool_rows, "error": error,
        "variance_raw": variance_raw, "remainder_raw": remainder_raw,
        "variance_r": variance_r, "remainder_r": error - variance_r,
        "variance_rn": variance_rn, "remainder_rn": error - variance_rn,
    }


def component_rows(sum_predicted, sum_log_predicted, y_test, size_runs, n, pool_rows, keys):
    """The rows of one size for dev_components.csv, one per configuration (stage 3 step 7).

    size_runs is the table of dev_runs.csv rows of that size; keys names the
    size: variant, year, split and size.
    """
    rows = []
    for i, configuration in enumerate(CONFIGURATIONS):
        deviance_per_draw = size_runs.loc[size_runs["config"] == configuration["config"], "test_deviance"].to_numpy()
        row = dict(keys)
        row.update(configuration)
        row.update(split_error(sum_predicted[i], sum_log_predicted[i], y_test, deviance_per_draw, n, pool_rows))
        rows.append(row)
    return rows


def run_experiment(X_pool, y_pool, pair_pool, X_test, y_test, variant, year, split_id, sizes, draws, noise=False,
                   report_ridge=True):
    """Every size and draw of one experiment, as stage 3 steps 4 to 7 describe them (section 6).

    Stage 3 runs the base experiment with it, and stage 6 the checks noise and
    split2. For each size, every draw is run (run_draws), its predictions are
    checked to be positive and finite, its rows kept for dev_runs.csv
    (draw_rows), and the running sums of the predictions and of their logs
    kept over the draws. The predictions themselves are not saved. After the
    last draw of a size, the sums give the split of the error into variance
    and remainder, one row per configuration (component_rows).

    After each size it prints the checks of section 8 on ppml: the fits that
    fell back to L-BFGS, the draws with a constant indicator, and the largest
    relative difference between the predicted and the observed training total.
    With report_ridge, it also prints, for each penalty of the grid, in how
    many draws the ridge line chose it (stage 3; stage 6 reports nothing on
    the ridge line, decision 18). Returns the two tables, with the columns of
    dev_runs.csv and dev_components.csv.
    """
    groups = test_groups(y_test)
    runs, components = [], []
    for n in sizes:
        start = time.perf_counter()
        sum_predicted = np.zeros((len(CONFIGURATIONS), len(y_test)))
        sum_log_predicted = np.zeros((len(CONFIGURATIONS), len(y_test)))
        size_rows = []
        for draw, result in enumerate(run_draws(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draws,
                                                noise)):
            keys = {"variant": variant, "year": year, "split": split_id, "size": n, "draw": draw}
            check_predictions(result["test_predicted"], keys)
            sum_predicted += result["test_predicted"]
            sum_log_predicted += np.log(result["test_predicted"])
            size_rows.extend(draw_rows(result, y_test, groups, keys))
        size_runs = runs_table(size_rows)
        size_keys = {"variant": variant, "year": year, "split": split_id, "size": n}
        components.extend(component_rows(sum_predicted, sum_log_predicted, y_test, size_runs, n, len(y_pool),
                                         size_keys))
        runs.append(size_runs)

        ppml = size_runs[size_runs["config"] == "ppml"]
        print(f"n = {n:>5,}: {draws} draws in {(time.perf_counter() - start) / 60:.1f} minutes. "
              f"ppml: {int(ppml['lbfgs_fallback'].sum())} fit(s) fell back to L-BFGS, "
              f"{int((ppml['constant_columns'] > 0).sum())} draw(s) with a constant indicator, "
              f"largest relative difference of the training total {(ppml['train_total_ratio'] - 1).abs().max():.1e}")
        if FIT_RIDGE and report_ridge:
            chosen = size_runs.loc[size_runs["config"] == "ridge", "ridge_alpha"]
            print("           ridge: draws in which cross-validation chose each penalty:")
            for alpha in RIDGE_ALPHAS:
                print(f"             {alpha:>7}: {int((chosen == alpha).sum())}")
    return pd.concat(runs, ignore_index=True), components_table(components)


def largest_relative_difference(new, old, keys, skip=("seconds",)):
    """The largest relative difference between two tables over the rows and columns they share.

    Rows are matched on keys. For a number, the difference is |a - b| divided
    by the larger of |a| and |b| (0 when the two are equal or both missing,
    and infinite when only one is missing or only one is infinite). Any
    difference in a text column is infinite. Returns the difference and the
    number of rows shared.
    """
    shared_columns = [column for column in new.columns if column in old.columns and column not in keys
                      and column not in skip]
    merged = new.merge(old, on=keys, suffixes=("_new", "_old"))
    largest = 0.0
    for column in shared_columns:
        a, b = merged[f"{column}_new"], merged[f"{column}_old"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a = a.to_numpy(dtype=float, na_value=np.nan)
            b = b.to_numpy(dtype=float, na_value=np.nan)
            both_missing = np.isnan(a) & np.isnan(b)
            one_missing = np.isnan(a) ^ np.isnan(b)
            scale = np.maximum(np.abs(a), np.abs(b))
            with np.errstate(invalid="ignore", divide="ignore"):
                difference = np.where(scale > 0, np.abs(a - b) / scale, 0.0)
            # Equal values, infinities included, do not differ; any other case
            # that gives no number (an infinity against a finite value) is a difference.
            difference[a == b] = 0.0
            difference[np.isnan(difference)] = np.inf
            difference[both_missing] = 0.0
            difference[one_missing] = np.inf
            if len(difference):
                largest = max(largest, float(difference.max()))
        elif not (a.fillna("").astype(str) == b.fillna("").astype(str)).all():
            largest = np.inf
    return largest, len(merged)


# ---------------------------------------------------------------------------
# Reading the draws: shared by the checkpoint (03b) and stages 4, 5 and 7
# ---------------------------------------------------------------------------

# The lines of the learning curves and of the checkpoint tables: the three
# candidates (rf and gbm at their tuned setting) and the reference lines mean
# and tree. Part 1 has no ridge rows, so the checkpoint leaves out the third
# reference line, ridge (decision 20); stage 7 adds it to its learning curves.
CANDIDATES = ("ppml", "rf", "gbm")
LINES = CANDIDATES + ("mean", "tree")
SETTING_COLUMNS = ["rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees"]


def summarise(runs):
    """For every size and configuration, the test deviance summarised over draws (checkpoint item 2, stage 4 step 1).

    The standard error is the standard deviation over draws divided by the
    square root of the number of draws. largest_share is the share of the sum
    over draws that comes from the largest draw: where it is over one half, one
    draw holds more than half of the mean, and the standard error is marked
    unreliable.
    """
    grouped = runs.groupby(["size", "config"], sort=True)
    summary = grouped.agg(
        model=("model", "first"),
        draws=("test_deviance", "size"),
        mean=("test_deviance", "mean"),
        sd=("test_deviance", "std"),
        median=("test_deviance", "median"),
        largest=("test_deviance", "max"),
        total=("test_deviance", "sum"),
        d2_mean=("d2", "mean"),
        d2_median=("d2", "median"),
        mae_rel=("mae_rel", "mean"),
        rmse_rel=("rmse_rel", "mean"),
        train_deviance=("train_deviance", "mean"),
    ).reset_index()
    settings = grouped[SETTING_COLUMNS].first().reset_index()
    summary = summary.merge(settings, on=["size", "config"])
    summary["se"] = summary["sd"] / np.sqrt(summary["draws"])
    summary["largest_share"] = summary["largest"] / summary["total"]
    summary["se_unreliable"] = summary["largest_share"] > 0.5
    return summary


def tune(summary):
    """The tuned configuration of rf and of gbm at each size: the lowest mean test deviance (stage 4 step 2).

    A tie (equal means, as for every gbm depth at n = 50, where a leaf of 20
    allows one split) goes to the simpler setting: for gbm the smaller depth,
    then fewer trees; for rf the larger min_samples_leaf, then fewer predictors
    per split.
    """
    rf = summary[summary["model"] == "rf"].sort_values(
        ["size", "mean", "rf_min_leaf", "rf_features"], ascending=[True, True, False, True])
    gbm = summary[summary["model"] == "gbm"].sort_values(
        ["size", "mean", "gbm_depth", "gbm_trees"], ascending=[True, True, True, True])
    tuned = pd.concat([rf.groupby("size").head(1), gbm.groupby("size").head(1)])
    return tuned.sort_values(["size", "model"]).reset_index(drop=True)


def line_configs(tuned):
    """The configuration each line follows at each size: ppml, rf and gbm at their tuned setting, mean and tree."""
    rows = []
    for size in sorted(tuned["size"].unique()):
        for line in LINES:
            if line in ("rf", "gbm"):
                at_size = tuned[(tuned["size"] == size) & (tuned["model"] == line)]
                config = at_size["config"].item()
            else:
                config = line
            rows.append({"size": size, "line": line, "config": config})
    return pd.DataFrame(rows)


def per_draw(runs, lines, column="test_deviance"):
    """One row per size and draw, one column per line, holding that line's value of column on the draw."""
    merged = runs.merge(lines, on=["size", "config"])
    wide = merged.pivot(index=["size", "draw"], columns="line", values=column).reset_index()
    wide.columns.name = None
    return wide


def learning_curves(summary, lines):
    """For every size and line: the mean test deviance, a band of two standard errors and the mean training deviance.

    Checkpoint item 1, and the figure learning_curves of stage 7. The training
    deviance follows the tuned setting at each size.
    """
    curves = summary.merge(lines, on=["size", "config"])
    curves = curves[["size", "line", "config", "draws", "mean", "se", "train_deviance"]].copy()
    curves["lower"] = curves["mean"] - 2 * curves["se"]
    curves["upper"] = curves["mean"] + 2 * curves["se"]
    return curves.sort_values(["line", "size"]).reset_index(drop=True)


def ppml_wins(draws, best):
    """By size, the number of draws in which ppml has the lower test deviance than tuned rf, tuned gbm and best.

    draws comes from per_draw, best from better_ensemble. Used by the checkpoint
    (item 3) and stage 4 step 1 (decision 12). A tie is not a win.
    """
    merged = draws.merge(best, on="size")
    best_deviance = np.where(merged["best"] == "rf", merged["rf"], merged["gbm"])
    merged["beats_rf"] = merged["ppml"] < merged["rf"]
    merged["beats_gbm"] = merged["ppml"] < merged["gbm"]
    merged["beats_best"] = merged["ppml"] < best_deviance
    wins = merged.groupby("size").agg(draws=("draw", "size"), best=("best", "first"),
                                      ppml_beats_rf=("beats_rf", "sum"), ppml_beats_gbm=("beats_gbm", "sum"),
                                      ppml_beats_best=("beats_best", "sum"))
    return wins.reset_index()


def selection_choices(draws, picks):
    """One row per size and draw: the model the size rule picked, the winner of the draw, and the regret.

    draws comes from per_draw, picks from size_rule. The regret (difference)
    is the picked model's test deviance minus the lowest of the three on the
    same draw. The result is what selection_confusion reads.
    """
    choices = draws.merge(picks.rename(columns={"chosen": "picked"}), on="size")
    choices["winner"] = winners(choices)
    choices["picked_deviance"] = np.nan
    for model in CANDIDATES:
        is_picked = choices["picked"] == model
        choices.loc[is_picked, "picked_deviance"] = choices.loc[is_picked, model]
    choices["difference"] = choices["picked_deviance"] - choices[list(CANDIDATES)].min(axis=1)
    return choices


def size_rule(summary, lines):
    """At each size, the candidate with the lowest mean test deviance (stage 4 step 3).

    The candidates are ppml and rf and gbm at their tuned setting. A tie goes
    to the first of ppml, rf, gbm.
    """
    means = summary.merge(lines, on=["size", "config"])
    rows = []
    for size, at_size in means.groupby("size"):
        values = at_size.set_index("line")["mean"]
        chosen = CANDIDATES[0]
        for line in CANDIDATES[1:]:
            # Strictly lower only, so that a tie stays with the earlier candidate.
            if values[line] < values[chosen]:
                chosen = line
        rows.append({"size": size, "chosen": chosen})
    return pd.DataFrame(rows)


def better_ensemble(tuned):
    """At each size, the better of tuned rf and tuned gbm by mean test deviance (the best comparison).

    A tie goes to rf, the first of the two.
    """
    rows = []
    for size, at_size in tuned.groupby("size"):
        rf_mean = at_size.loc[at_size["model"] == "rf", "mean"].item()
        gbm_mean = at_size.loc[at_size["model"] == "gbm", "mean"].item()
        rows.append({"size": size, "best": "rf" if rf_mean <= gbm_mean else "gbm"})
    return pd.DataFrame(rows)


def winners(draws, candidates=CANDIDATES):
    """The candidate with the lowest test deviance on each draw. A tie goes to the first of ppml, rf, gbm."""
    values = draws[list(candidates)].to_numpy()
    return np.array(candidates)[np.argmin(values, axis=1)]


def paired_se(differences):
    """The standard error of a mean of paired differences: their standard deviation over draws / sqrt(draws)."""
    differences = np.asarray(differences, dtype=float)
    return float(np.std(differences, ddof=1) / np.sqrt(len(differences)))


def selection_confusion(choices, candidates=CANDIDATES):
    """The model a selector picked against the model that won, per size and pooled over sizes.

    choices has one row per size and draw, with the columns picked (the model
    the selector picked), winner (the model with the lowest test deviance on
    that draw) and difference (the picked model's test deviance minus the
    winner's). Used by the checkpoint (item 3) and by stage 5
    (selection_confusion.csv).

    Returns two tables. counts: for each size and for all sizes together, the
    number of draws for every picked model and winner. scores: for each size
    and for all sizes, the share picked right, the share that always picking
    the most frequent winner of each size would get, and the mean difference
    with its paired standard error.
    """
    count_rows, score_rows = [], []
    scopes = [(str(size), group) for size, group in choices.groupby("size")] + [("all", choices)]
    for scope, group in scopes:
        for picked in candidates:
            picked_rows = group[group["picked"] == picked]
            # At one size the size rule picks one model, so its table is one row;
            # pooled over sizes the table keeps all three rows, zeros included.
            if len(picked_rows) == 0 and scope != "all":
                continue
            row = {"sizes": scope, "picked": picked, "draws": len(picked_rows)}
            for winner in candidates:
                row[f"won_{winner}"] = int((picked_rows["winner"] == winner).sum())
            count_rows.append(row)
        # The best a selector that picks one model per size can do: the most frequent winner of each size.
        most_frequent = sum(at_size["winner"].value_counts().max() for _, at_size in group.groupby("size"))
        score_rows.append({
            "sizes": scope,
            "draws": len(group),
            "share_picked_right": float((group["picked"] == group["winner"]).mean()),
            "share_most_frequent_winner": most_frequent / len(group),
            "mean_difference": float(group["difference"].mean()),
            "se_difference": paired_se(group["difference"]),
        })
    return pd.DataFrame(count_rows), pd.DataFrame(score_rows)


def find_crossing(sizes, gap):
    """The pattern of the error gap E(ppml) - E(c) over the sizes (section 7, decision 12).

    Each gap is read as negative (ppml ahead) or as zero or positive.
    never_overtaken: all negative. never_ahead: all zero or positive.
    one_crossing: one change, from negative to zero or positive; n_star is
    interpolated linearly in log n between the two sizes around it.
    reversal: one change, from zero or positive to negative; no n_star.
    reversal_then_crossing: two changes, first to negative and then back;
    last_crossing is interpolated at the second change as n_star is.
    several: anything else, with no n_star. crossing_from and crossing_to are
    the two sizes around the change that n_star or last_crossing comes from.
    """
    sizes = np.asarray(sizes, dtype=float)
    gap = np.asarray(gap, dtype=float)
    negative = gap < 0
    changes = [i for i in range(len(gap) - 1) if negative[i] != negative[i + 1]]
    result = {"pattern": "several", "n_star": np.nan, "last_crossing": np.nan,
              "crossing_from": np.nan, "crossing_to": np.nan}
    if not changes:
        result["pattern"] = "never_overtaken" if negative.all() else "never_ahead"
    elif len(changes) == 1 and negative[changes[0]]:
        result["pattern"] = "one_crossing"
        result["n_star"] = interpolate_crossing(sizes, gap, changes[0])
        result["crossing_from"], result["crossing_to"] = int(sizes[changes[0]]), int(sizes[changes[0] + 1])
    elif len(changes) == 1:
        result["pattern"] = "reversal"
    elif len(changes) == 2 and not negative[changes[0]]:
        result["pattern"] = "reversal_then_crossing"
        result["last_crossing"] = interpolate_crossing(sizes, gap, changes[1])
        result["crossing_from"], result["crossing_to"] = int(sizes[changes[1]]), int(sizes[changes[1] + 1])
    return result


def interpolate_crossing(sizes, gap, i):
    """The size where the gap reaches zero between sizes[i] and sizes[i + 1], linear in log n."""
    share = (0 - gap[i]) / (gap[i + 1] - gap[i])
    log_size = np.log(sizes[i]) + share * (np.log(sizes[i + 1]) - np.log(sizes[i]))
    return float(np.exp(log_size))


def check_find_crossing():
    """Section 8: find_crossing on toy gaps at the sizes 50, 100, 200 and 500."""
    sizes = (50, 100, 200, 500)
    one = find_crossing(sizes, (-2, -1, 1, 2))
    assert one["pattern"] == "one_crossing" and round(one["n_star"], 2) == 141.42, one
    assert (one["crossing_from"], one["crossing_to"]) == (100, 200), one
    back = find_crossing(sizes, (2, -1, 1, 2))
    assert back["pattern"] == "reversal_then_crossing" and round(back["last_crossing"], 2) == 141.42, back
    assert find_crossing(sizes, (1, -1, -2, -3))["pattern"] == "reversal"
    assert find_crossing(sizes, (-1, -2, -3, -4))["pattern"] == "never_overtaken"
    assert find_crossing(sizes, (1, 2, 3, 4))["pattern"] == "never_ahead"
    assert find_crossing(sizes, (-1, 1, -1, 1))["pattern"] == "several"


def gaps(components, runs, tuned):
    """The error, bias and variance gaps of ppml against tuned rf, tuned gbm and the better of the two (stage 4 step 4).

    error gap = E(ppml) - E(c). For each version v of section 7 (raw, r, rn):
    bias gap = B_v(ppml) - B_v(c) and variance gap = V_v(c) - V_v(ppml), where B
    is the remainder, so error gap = bias gap - variance gap. The standard
    error of the error gap is paired over draws.
    """
    best = better_ensemble(tuned)
    rows = []
    for size in sorted(tuned["size"].unique()):
        at_size = components[components["size"] == size].set_index("config")
        ppml = at_size.loc["ppml"]
        best_model = best.loc[best["size"] == size, "best"].item()
        for comparison in ("rf", "gbm", "best"):
            model = best_model if comparison == "best" else comparison
            config = tuned.loc[(tuned["size"] == size) & (tuned["model"] == model), "config"].item()
            other = at_size.loc[config]
            ppml_draws = runs[(runs["size"] == size) & (runs["config"] == "ppml")].set_index("draw")["test_deviance"]
            other_draws = runs[(runs["size"] == size) & (runs["config"] == config)].set_index("draw")["test_deviance"]
            differences = (ppml_draws - other_draws).dropna()
            row = {"size": size, "comparison": comparison, "model": model, "config": config,
                   "error_ppml": ppml["error"], "error_other": other["error"],
                   "error_gap": ppml["error"] - other["error"], "se_error_gap": paired_se(differences)}
            row["within_2se"] = row["error_gap"] <= 2 * row["se_error_gap"]
            for version in ("raw", "r", "rn"):
                row[f"bias_gap_{version}"] = ppml[f"remainder_{version}"] - other[f"remainder_{version}"]
                row[f"variance_gap_{version}"] = other[f"variance_{version}"] - ppml[f"variance_{version}"]
                # Section 8: error gap = bias gap - variance gap, to a relative tolerance of 1e-8 of the errors.
                identity = row[f"bias_gap_{version}"] - row[f"variance_gap_{version}"]
                scale = max(abs(ppml["error"]), abs(other["error"]))
                assert abs(row["error_gap"] - identity) <= 1e-8 * scale, \
                    f"error gap is not bias gap - variance gap at size {size}, {comparison}, {version}."
            rows.append(row)
    return pd.DataFrame(rows)


def crossings(gap_table):
    """find_crossing on the error gap of each comparison, over the sizes in the table."""
    rows = []
    for comparison in ("rf", "gbm", "best"):
        rows_of_comparison = gap_table[gap_table["comparison"] == comparison].sort_values("size")
        result = find_crossing(rows_of_comparison["size"], rows_of_comparison["error_gap"])
        rows.append({"comparison": comparison, "sizes": len(rows_of_comparison), **result})
    return pd.DataFrame(rows)


# Grid edges (section 6, Checkpoint): the step beyond the first and the last
# value of each setting's grid, fixed before part 1. None means the grid cannot
# be extended on that side. The rule has one round, read on the first run of
# part 1, and decision 19 applied it. On the rerun of part 1 and later, an edge
# on a side that could be extended is reported as a limit of the grid, with no
# step.
EDGE_STEPS = {
    ("rf", "rf_features"): {"below": "2 predictors per split", "above": None},
    ("rf", "rf_min_leaf"): {"below": None, "above": "min_samples_leaf 50"},
    ("gbm", "gbm_depth"): {"below": None, "above": "depth 7"},
    ("gbm", "gbm_trees"): {"below": "10 trees (staged, no extra fit)",
                           "above": "750 and 1,000 trees, with GBM_MAX_TREES 1,000"},
}
CANNOT_EXTEND = "cannot be extended"
GRID_LIMIT = "limit of the grid (the one round of Grid edges is done, decision 19)"


def edges(runs, summary, tuned, first_round):
    """Whether each tuned setting sits at an edge of its grid (checkpoint item 5, rule Grid edges).

    A tuned setting is at an edge when its value is the first or the last of
    that setting's grid, and its mean test deviance is lower than that of the
    configuration with the next value inside the grid, the model's other
    setting held at its tuned value, by more than twice the paired standard
    error. A tie is never an edge.

    Each grid is the one the inputs were made on (the values in summary), not
    the one in the parameters above, so the report on the first run of part 1
    reads the same after decision 19 widened the grid. first_round is True only
    for the first run of part 1, the one round of the rule: only then does an
    edge get its step beyond.
    """
    rows = []
    for _, chosen in tuned.iterrows():
        for (model, setting), steps in EDGE_STEPS.items():
            if chosen["model"] != model:
                continue
            grid = sorted(int(value) for value in summary.loc[summary["model"] == model, setting].dropna().unique())
            value = int(chosen[setting])
            if value not in (grid[0], grid[-1]) or len(grid) < 2:
                continue
            side = "below" if value == grid[0] else "above"
            neighbour_value = grid[1] if side == "below" else grid[-2]
            # The neighbour: the same model at the same size, with this setting
            # at the next value inside the grid and every other setting as tuned.
            same = summary[(summary["size"] == chosen["size"]) & (summary["model"] == model)]
            for other_setting in SETTING_COLUMNS:
                if other_setting.startswith(model) and other_setting != setting:
                    same = same[same[other_setting] == chosen[other_setting]]
            neighbour = same[same[setting] == neighbour_value]
            at_size = runs[runs["size"] == chosen["size"]]
            chosen_draws = at_size[at_size["config"] == chosen["config"]].set_index("draw")["test_deviance"]
            neighbour_draws = at_size[at_size["config"] == neighbour["config"].item()].set_index("draw")["test_deviance"]
            differences = (neighbour_draws - chosen_draws).dropna()
            difference = float(differences.mean())
            se = paired_se(differences)
            at_edge = difference > 2 * se
            rows.append({
                "size": chosen["size"], "model": model, "setting": setting, "tuned_value": value,
                "side": side, "neighbour_value": neighbour_value, "neighbour_config": neighbour["config"].item(),
                "neighbour_minus_tuned": difference, "se": se, "at_edge": at_edge,
                "extension": edge_extension(at_edge, steps[side], first_round),
            })
    # The columns are named even when no tuned setting is at either end of its grid.
    columns = ["size", "model", "setting", "tuned_value", "side", "neighbour_value", "neighbour_config",
               "neighbour_minus_tuned", "se", "at_edge", "extension"]
    return pd.DataFrame(rows, columns=columns)


def edge_extension(at_edge, step, first_round):
    """What the last column of item 5 says about an edge: its step, that the grid cannot be extended, or a limit."""
    if not at_edge:
        return ""
    if step is None:
        return CANNOT_EXTEND
    return step if first_round else GRID_LIMIT


def deviance_by_group(runs, lines):
    """For each line and size, the three group columns averaged over draws, and each group's share of their sum (item 6)."""
    merged = runs.merge(lines, on=["size", "config"])
    columns = [f"deviance_{name}" for name in GROUP_NAMES]
    table = merged.groupby(["size", "line", "config"])[columns].mean().reset_index()
    total = table[columns].sum(axis=1)
    for name in GROUP_NAMES:
        table[f"share_{name}"] = table[f"deviance_{name}"] / total
    return table


def worse_than_mean(runs):
    """Every configuration and draw whose test deviance is above that of mean on the same sample (item 7)."""
    mean_draws = runs[runs["config"] == "mean"][["size", "draw", "test_deviance"]]
    merged = runs.merge(mean_draws, on=["size", "draw"], suffixes=("", "_mean"))
    worse = merged[(merged["test_deviance"] > merged["test_deviance_mean"]) & (merged["config"] != "mean")]
    return worse[["size", "draw", "model", "config", "test_deviance", "test_deviance_mean", "max_pred_ratio"]]


def divergences(runs):
    """gbm draws that diverge as in decision 9 (rule Stability).

    A divergence is a gbm configuration on a draw whose test deviance is more
    than ten times that of mean on the same sample, or whose largest test
    prediction is more than 1,000 times the largest training flow.
    """
    mean_draws = runs[runs["config"] == "mean"][["size", "draw", "test_deviance"]]
    merged = runs[runs["model"] == "gbm"].merge(mean_draws, on=["size", "draw"], suffixes=("", "_mean"))
    diverged = (merged["test_deviance"] > 10 * merged["test_deviance_mean"]) | (merged["max_pred_ratio"] > 1000)
    return merged[diverged][["size", "draw", "config", "test_deviance", "test_deviance_mean", "max_pred_ratio"]]


def health_table(runs, components):
    """By size, the health checks of checkpoint item 7 in one table.

    The largest relative difference in the identity error = remainder_raw +
    variance_raw (section 8); for ppml, the draws with an indicator constant
    in the sample, the fits scikit-learn warned had fallen back to L-BFGS and
    the largest relative difference between the predicted and the observed
    training total; the rows worse than mean on the same sample; and the gbm
    divergences of the rule Stability.
    """
    identity = (components["error"] - components["remainder_raw"] - components["variance_raw"]).abs() / components["error"]
    worse = worse_than_mean(runs)
    diverged = divergences(runs)
    rows = []
    for n in sorted(runs["size"].unique()):
        ppml = runs[(runs["size"] == n) & (runs["config"] == "ppml")]
        rows.append({
            "size": n,
            "identity_largest_difference": identity[components["size"] == n].max(),
            "ppml_draws_with_constant_indicator": int((ppml["constant_columns"] > 0).sum()),
            "ppml_lbfgs_fallbacks": int(ppml["lbfgs_fallback"].sum()),
            "ppml_largest_total_difference": (ppml["train_total_ratio"] - 1).abs().max(),
            "rows_worse_than_mean": int((worse["size"] == n).sum()),
            "gbm_divergences": int((diverged["size"] == n).sum()),
        })
    return pd.DataFrame(rows)


def seconds_per_draw(runs):
    """Seconds per draw, summed over the fits of the draw (item 8), and the seconds of the gbm fits alone.

    A gbm fit gives every tree count of its depth, so each depth is counted
    once, not once per tree count.
    """
    fits = runs.copy()
    fits["fit"] = np.where(fits["model"] == "gbm", "gbm_depth" + fits["gbm_depth"].astype(str), fits["config"])
    fits = fits.drop_duplicates(["size", "draw", "fit"])
    per_draw_seconds = fits.groupby(["size", "draw"])["seconds"].sum()
    gbm_seconds = fits[fits["model"] == "gbm"].groupby(["size", "draw"])["seconds"].sum()
    table = pd.DataFrame({"seconds": per_draw_seconds, "gbm_seconds": gbm_seconds}).reset_index()
    return table.groupby("size")[["seconds", "gbm_seconds"]].mean().reset_index()


# Section 9 of the build plan: seconds per draw (13 fits, 37 configurations)
# that Claude measured on 6 October 2026 on one core of a sandbox, on a
# stand-in of the CEPII data, and the minutes it projected from them for
# stages 5 and 6. Projections scale these by the seconds measured here.
SECTION9_DRAW_SECONDS = {50: 2.4, 1000: 2.5, 5000: 7.3}
SECTION9_STAGE_MINUTES = {"05_analysis.ipynb": 70, "06_checks.ipynb": 40}


def section9_seconds(n):
    """Section 9's seconds per draw at size n, linear in n between the sizes it measured."""
    measured = sorted(SECTION9_DRAW_SECONDS)
    return float(np.interp(n, measured, [SECTION9_DRAW_SECONDS[size] for size in measured]))


def projected_seconds(seconds_at, size_measured, n):
    """Seconds per draw at size n, scaled from seconds_at measured at size_measured by the ratios of section 9."""
    return seconds_at * section9_seconds(n) / section9_seconds(size_measured)


def ppml_coefficients(runs):
    """By size, the spread over draws of each ppml coefficient, and the draws where its column was dropped (item 9)."""
    ppml = runs[runs["config"] == "ppml"]
    rows = []
    for size, at_size in ppml.groupby("size"):
        for name in ["intercept"] + PREDICTORS:
            values = at_size[f"coef_{name}"].dropna()
            rows.append({
                "size": size, "coefficient": name, "draws_estimated": len(values),
                "draws_dropped": int(at_size[f"coef_{name}"].isna().sum()),
                "median": values.median(), "p10": values.quantile(0.1), "p90": values.quantile(0.9),
                "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows)


def thin_indicators(runs):
    """By size, for each indicator, the draws in which it had one to three 1s in the training sample (item 9)."""
    ppml = runs[runs["config"] == "ppml"]
    rows = []
    for size, at_size in ppml.groupby("size"):
        for name in INDICATORS:
            ones = at_size[f"ones_{name}"]
            rows.append({"size": size, "indicator": name, "draws": len(ones),
                         "draws_with_no_1": int((ones == 0).sum()),
                         "draws_with_1_to_3": int(ones.between(1, 3).sum()),
                         "median_ones": ones.median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 4: the size rule, the gaps and the cutoff (section 6 of the plan)
# ---------------------------------------------------------------------------

# The three comparisons of stage 4 step 4, and the six patterns find_crossing
# can return (section 7, decision 12).
COMPARISONS = ("rf", "gbm", "best")
PATTERNS = ("one_crossing", "reversal", "reversal_then_crossing", "never_overtaken", "never_ahead", "several")
# The columns of size_rule.csv, in order (stage 4 step 3).
SIZE_RULE_COLUMNS = ["size", "model", "config", "rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees",
                     "draws", "mean", "se", "se_unreliable", "median", "chosen", "edges"]


def gbm_extremes(runs):
    """By size, how close gbm came to the two thresholds of the rule Stability, and how many gbm rows crossed one.

    The largest test deviance of a gbm configuration divided by that of mean on
    the same sample (the threshold is 10), the largest test prediction divided
    by the largest training flow (the threshold is 1,000), and the number of
    gbm rows (one configuration on one draw) that divergences flags. Stage 4
    step 0 reads it on the sizes above CHECKPOINT_MAX.
    """
    mean_draws = runs[runs["config"] == "mean"][["size", "draw", "test_deviance"]]
    gbm = runs[runs["model"] == "gbm"].merge(mean_draws, on=["size", "draw"], suffixes=("", "_mean"))
    gbm["ratio_to_mean"] = gbm["test_deviance"] / gbm["test_deviance_mean"]
    diverged = divergences(runs)
    rows = []
    for size, at_size in gbm.groupby("size"):
        rows.append({"size": size, "gbm_rows": len(at_size),
                     "largest_ratio_to_mean": at_size["ratio_to_mean"].max(),
                     "largest_max_pred_ratio": at_size["max_pred_ratio"].max(),
                     "divergences": int((diverged["size"] == size).sum())})
    # The columns are named so that a table with no size still has them.
    return pd.DataFrame(rows, columns=["size", "gbm_rows", "largest_ratio_to_mean", "largest_max_pred_ratio",
                                       "divergences"])


def edge_note(edge_table, size, model):
    """The settings of a tuned model that sit at an edge of their grid at one size, in words; empty if none."""
    at_size = edge_table[(edge_table["size"] == size) & (edge_table["model"] == model)]
    notes = [f"{row.setting} {row.side}: {row.extension}" for row in at_size.itertuples() if row.at_edge]
    return "; ".join(notes)


def size_rule_table(summary, tuned, runs):
    """size_rule.csv: at each size, ppml, tuned rf and tuned gbm, and the one the size rule chooses (stage 4 step 3).

    Each row has the model's tuned setting (none for ppml), its mean test
    deviance with its standard error and its median (step 1), se_unreliable
    (one draw holds more than half of the sum over draws, so the standard error
    is unreliable), and whether the size rule chooses it. For rf and gbm, edges
    lists the settings that sit at an edge of their grid as the checkpoint
    defines it (item 5). The one round
    of the rule Grid edges is done (decision 19), so an edge is reported, not
    extended: "cannot be extended" or "limit of the grid".
    """
    lines = line_configs(tuned)
    candidates = lines[lines["line"].isin(CANDIDATES)]
    table = candidates.merge(summary, on=["size", "config"])
    picks = size_rule(summary, lines).rename(columns={"chosen": "chosen_model"})
    table = table.merge(picks, on="size")
    table["chosen"] = table["model"] == table["chosen_model"]
    edge_table = edges(runs, summary, tuned, first_round=False)
    table["edges"] = [edge_note(edge_table, row.size, row.model) for row in table.itertuples()]
    # Rows in the order ppml, rf, gbm at each size.
    table["order"] = [CANDIDATES.index(model) for model in table["model"]]
    table = table.sort_values(["size", "order"]).reset_index(drop=True)
    table = table[SIZE_RULE_COLUMNS]
    for column in SETTING_COLUMNS:
        table[column] = table[column].astype("Int64")
    return table


def gaps_with_wins(gap_table, wins):
    """gaps.csv: the gaps of stage 4 step 4, with the draws in which ppml has the lower test deviance (step 1).

    wins comes from ppml_wins. For each comparison, ppml_wins is the number of
    draws, out of draws, in which ppml beats that comparison's ensemble on the
    same training sample (decision 12). A tie is not a win.
    """
    table = gap_table.merge(wins[["size", "draws", "ppml_beats_rf", "ppml_beats_gbm", "ppml_beats_best"]], on="size")
    # Each row takes the win count of its own comparison.
    table["ppml_wins"] = [row[f"ppml_beats_{row['comparison']}"] for _, row in table.iterrows()]
    return table.drop(columns=["ppml_beats_rf", "ppml_beats_gbm", "ppml_beats_best"])


def candidate_matrix(runs, n):
    """At size n, the test deviance of every candidate configuration on every draw (stage 4 step 6).

    Returns the matrix, one row per draw in the order of the draws and one
    column per configuration, and a table naming the model and configuration of
    each column. ppml comes first. Within rf and within gbm the columns are in
    the order of the tie-break of tune (rf: larger min_samples_leaf, then fewer
    predictors per split; gbm: smaller depth, then fewer trees), so the first
    column with the lowest mean is the one tune picks.
    """
    at_size = runs[(runs["size"] == n) & runs["model"].isin(CANDIDATES)]
    configs = at_size.drop_duplicates("config")[["model", "config"] + SETTING_COLUMNS]
    ppml = configs[configs["model"] == "ppml"]
    rf = configs[configs["model"] == "rf"].sort_values(["rf_min_leaf", "rf_features"], ascending=[False, True])
    gbm = configs[configs["model"] == "gbm"].sort_values(["gbm_depth", "gbm_trees"])
    columns = pd.concat([ppml, rf, gbm]).reset_index(drop=True)
    wide = at_size.pivot(index="draw", columns="config", values="test_deviance").sort_index()
    matrix = wide[list(columns["config"])].to_numpy()
    assert np.isfinite(matrix).all(), f"Some candidate is missing a draw at size {n}."
    return matrix, columns


def resample_gaps(matrices, positions):
    """The error gaps of ppml at every size, on one set of draws, with the ensembles tuned on those draws.

    matrices maps each size to what candidate_matrix returns. positions maps
    each size to the rows of its matrix to use: a resample of the draws, in
    which a draw can come more than once, or every draw once. At each size the
    mean test deviance of every configuration is taken over those rows, rf and
    gbm are tuned again (the lowest mean, ties as in tune), the better of the
    two is the ensemble of the best comparison (a tie goes to rf, as in
    better_ensemble), and the error gap is E(ppml) - E(c) for each comparison.
    """
    rows = []
    for n, (matrix, columns) in matrices.items():
        means = matrix[positions[n]].mean(axis=0)
        row = {"size": n}
        ppml_mean = means[np.flatnonzero(columns["model"] == "ppml")[0]]
        tuned_means = {}
        for model in ("rf", "gbm"):
            where = np.flatnonzero(columns["model"] == model)
            # argmin returns the first of equal values, so a tie goes to the simpler setting.
            column = where[np.argmin(means[where])]
            row[f"{model}_config"] = columns["config"][column]
            tuned_means[model] = means[column]
        row["best"] = "rf" if tuned_means["rf"] <= tuned_means["gbm"] else "gbm"
        tuned_means["best"] = tuned_means[row["best"]]
        for comparison in COMPARISONS:
            row[f"gap_{comparison}"] = ppml_mean - tuned_means[comparison]
        rows.append(row)
    return pd.DataFrame(rows)


def check_resampling(matrices, tuned, gap_table):
    """Before the bootstrap: on every draw once, resample_gaps must give the tuned settings and error gaps of steps 2 and 4.

    The error gaps are compared to a relative 1e-9 of the errors: the two add up
    the same draws in a different order, so they can differ in the last digits.
    """
    every_draw = {n: np.arange(len(matrix)) for n, (matrix, _) in matrices.items()}
    again = resample_gaps(matrices, every_draw)
    for row in again.itertuples():
        for model in ("rf", "gbm"):
            expected = tuned.loc[(tuned["size"] == row.size) & (tuned["model"] == model), "config"].item()
            assert getattr(row, f"{model}_config") == expected, \
                f"The bootstrap tunes {model} differently from tune at size {row.size}."
        for comparison in COMPARISONS:
            expected = gap_table.loc[(gap_table["size"] == row.size) & (gap_table["comparison"] == comparison)].iloc[0]
            scale = max(abs(expected["error_ppml"]), abs(expected["error_other"]))
            assert abs(getattr(row, f"gap_{comparison}") - expected["error_gap"]) <= 1e-9 * scale, \
                f"The bootstrap's error gap differs from gaps at size {row.size}, {comparison}."


def candidate_matrices(runs):
    """candidate_matrix at every size of runs, from the smallest size up."""
    sizes = sorted(int(n) for n in runs["size"].unique())
    return {n: candidate_matrix(runs, n) for n in sizes}


def bootstrap_crossings(matrices, year, split_id, resamples):
    """find_crossing on resampled draws, for each comparison (stage 4 step 6).

    matrices comes from candidate_matrices. In resample b, at each size n
    separately, the draws of that size are drawn again with replacement, as many
    as there are, with rng("bootstrap", year, split_id, n, b), and every
    configuration uses the same drawn draws. Inside the resample rf and gbm are
    tuned again and the error gaps recomputed (resample_gaps), then
    find_crossing reads them. The
    bias and variance gaps are not resampled: they need each draw's
    predictions, which are not saved.

    Returns one row per resample and comparison with its pattern, n_star and
    last_crossing.
    """
    rows = []
    for resample in range(resamples):
        positions = {}
        for n, (matrix, _) in matrices.items():
            draws = len(matrix)
            positions[n] = rng("bootstrap", year, split_id, n, resample).integers(0, draws, size=draws)
        resampled = resample_gaps(matrices, positions)
        for comparison in COMPARISONS:
            result = find_crossing(resampled["size"], resampled[f"gap_{comparison}"])
            rows.append({"resample": resample, "comparison": comparison, "pattern": result["pattern"],
                         "n_star": result["n_star"], "last_crossing": result["last_crossing"]})
    return pd.DataFrame(rows)


def interval(values):
    """The 2.5 and 97.5 percentiles of values, or two empty values if there are none."""
    if len(values) == 0:
        return np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def cutoff_table(crossing_table, resampled):
    """cutoff.csv: the pattern and n* of each comparison (stage 4 step 5), with the bootstrap of step 6 beside them.

    crossing_table comes from crossings, resampled from bootstrap_crossings.
    For each comparison: the share of resamples in each pattern; the 2.5 and
    97.5 percentiles of n_star over the resamples with one_crossing, and of
    last_crossing over the resamples with reversal_then_crossing (decision 12),
    each with the number of resamples it rests on.
    """
    rows = []
    for _, point in crossing_table.iterrows():
        of_comparison = resampled[resampled["comparison"] == point["comparison"]]
        row = point.to_dict()
        row["resamples"] = len(of_comparison)
        for pattern in PATTERNS:
            row[f"share_{pattern}"] = float((of_comparison["pattern"] == pattern).mean())
        one = of_comparison.loc[of_comparison["pattern"] == "one_crossing", "n_star"]
        row["resamples_one_crossing"] = len(one)
        row["n_star_low"], row["n_star_high"] = interval(one)
        back = of_comparison.loc[of_comparison["pattern"] == "reversal_then_crossing", "last_crossing"]
        row["resamples_reversal_then_crossing"] = len(back)
        row["last_crossing_low"], row["last_crossing_high"] = interval(back)
        rows.append(row)
    table = pd.DataFrame(rows)
    for column in ["crossing_from", "crossing_to"]:
        table[column] = table[column].astype("Int64")
    return table


# ---------------------------------------------------------------------------
# Stage 5: the analysis year at the settings frozen after stage 4 (section 6)
# ---------------------------------------------------------------------------

# The methods of stage 5, in the order of their rows on each draw (stage 5 steps 1 to 5).
METHODS = ("ppml", "rf", "gbm", "size_rule", "best_of_three", "cv_select", "ridge")
# The columns of analysis_runs.csv, in order. constant_columns, lbfgs_fallback
# and train_total_ratio are filled on the ppml rows only, ridge_alpha on the
# ridge rows only.
ANALYSIS_RUN_COLUMNS = ["year", "split", "size", "draw", "method", "model", "config",
                        "rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees",
                        "test_deviance", "d2", "mae_rel", "rmse_rel", "seconds", "fits",
                        "constant_columns", "lbfgs_fallback", "train_total_ratio", "ridge_alpha"]
ANALYSIS_WHOLE_NUMBER_COLUMNS = ["rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees", "fits",
                                 "constant_columns", "lbfgs_fallback"]
# The columns of analysis_summary.csv, in order. The columns after fits are
# filled on one method's rows only: best and the win counts on ppml, the
# differences to cv_select (the test of H2) and to best_of_three on size_rule,
# and the difference to ppml and the win count of ridge on ridge.
SUMMARY_COLUMNS = ["size", "method", "model", "config", "draws", "mean", "se", "se_unreliable", "median",
                   "seconds", "fits",
                   "best", "ppml_beats_rf", "ppml_beats_gbm", "ppml_beats_best",
                   "minus_cv_select", "se_minus_cv_select", "draws_with_cv_select", "within_2se",
                   "minus_best_of_three", "se_minus_best_of_three",
                   "minus_ppml", "se_minus_ppml", "ridge_beats_best"]
# The two rows of analysis_cutoff.csv: each compares one model with the better
# of rf and gbm. best is the test of H1; ridge_best is decision 22, reported
# beside it and not part of it.
CUTOFF_COMPARISONS = {"best": "ppml", "ridge_best": "ridge"}
# The models whose draws the bootstrap of analysis_cutoff.csv resamples.
CUTOFF_MODELS = ("ppml", "rf", "gbm", "ridge")
# The fits of one draw of cv_select: on each fold one ppml fit, one fit per rf
# setting and one gbm fit per depth, then the refit (61 with decision 19). The
# ridge line has one fit per fold and penalty, then the refit (31).
CV_SELECT_FITS = CV_FOLDS * (1 + len(RF_MAX_FEATURES) * len(RF_MIN_LEAF) + len(GBM_DEPTHS)) + 1
RIDGE_FITS = CV_FOLDS * len(RIDGE_ALPHAS) + 1


def size_rule_lookup(table, n):
    """The rows of size_rule.csv for the grid size nearest to n on a log scale (section 7).

    Stage 5 runs the sizes of the grid itself, so the nearest size is n. If n
    were exactly halfway between two sizes on the log scale, the smaller would
    be taken.
    """
    grid = np.array(sorted(table["size"].unique()), dtype=float)
    nearest = grid[np.argmin(np.abs(np.log(grid) - np.log(n)))]
    return table[table["size"] == nearest]


def rf_max_features(features):
    """The value of RF_MAX_FEATURES that gives features predictors per split (rf_features in the files: 8 is 1.0)."""
    for value in RF_MAX_FEATURES:
        if rf_features(value) == features:
            return value
    raise ValueError(f"No value of RF_MAX_FEATURES gives {features} predictors per split.")


def frozen_settings(rows):
    """The settings of ppml, rf and gbm at one size, and the model the size rule chooses, from rows of size_rule.csv.

    Returns a dictionary with one entry per model (its model, config and
    settings, whole numbers where the setting applies) and the entry chosen.
    """
    settings = {}
    for row in rows.itertuples():
        setting = {"model": row.model, "config": row.config}
        for column in SETTING_COLUMNS:
            value = getattr(row, column)
            if not pd.isna(value):
                setting[column] = int(value)
        settings[row.model] = setting
    chosen = rows.loc[rows["chosen"], "model"]
    assert len(chosen) == 1, "size_rule.csv does not choose exactly one model at this size."
    settings["chosen"] = chosen.item()
    return settings


def candidate_settings():
    """The candidate configurations as settings, in the order cv_select breaks a tie (stage 5 step 4).

    ppml first, then rf from the larger min_samples_leaf and then the fewer
    predictors per split, then gbm from the smaller depth and then the fewer
    trees: the simpler setting first within each model, as in tune, and across
    models the order of ppml, rf, gbm, as in size_rule.
    """
    ppml = [setting for setting in CONFIGURATIONS if setting["model"] == "ppml"]
    rf = [setting for setting in CONFIGURATIONS if setting["model"] == "rf"]
    gbm = [setting for setting in CONFIGURATIONS if setting["model"] == "gbm"]
    # Each setting is sorted on a pair of numbers; the minus sign puts the larger leaf first.
    rf = sorted(rf, key=lambda setting: (-setting["rf_min_leaf"], setting["rf_features"]))
    gbm = sorted(gbm, key=lambda setting: (setting["gbm_depth"], setting["gbm_trees"]))
    return ppml + rf + gbm


def fit_setting(setting, X_train, y_train, X_other, seed):
    """Fit one candidate at one setting on X_train, y_train and predict the rows X_other: one fit.

    gbm is fitted with the number of trees of its setting, rf without
    out-of-bag predictions. Returns the predictions, the seconds of the fit
    call and, for ppml, what fit_ppml returns.
    """
    if setting["model"] == "ppml":
        ppml = fit_ppml(X_train, y_train, X_other)
        return {"predicted": ppml["test_predicted"], "seconds": ppml["seconds"], "ppml": ppml}
    if setting["model"] == "rf":
        model = make_rf(rf_max_features(setting["rf_features"]), setting["rf_min_leaf"], seed, oob=False)
    else:
        model = make_gbm(setting["gbm_depth"], seed, trees=setting["gbm_trees"])
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    return {"predicted": model.predict(X_other), "seconds": seconds, "ppml": None}


def validation_deviance(y, predicted, where):
    """The mean Poisson deviance on a validation fold, after checking that every prediction is positive and finite."""
    assert np.isfinite(predicted).all() and (predicted > 0).all(), \
        f"A prediction on a validation fold is not positive and finite at {where}. Stop and report."
    return mean_poisson_deviance(y, predicted)


def cv_select(X_train, y_train, pair_train, X_test, year, split_id, n, draw, seed):
    """Cross-validated selection over the candidate configurations on one training sample (stage 5 step 4).

    The sample is cut into CV_FOLDS folds that keep both directions of a pair
    together, the same folds as the ridge line (cv_folds). On each fold every
    candidate is fitted on the other folds and scored by its mean Poisson
    deviance on the fold: 1 ppml fit, one rf fit for each of its settings, and
    one gbm fit for each depth, with GBM_MAX_TREES trees, scored at every count
    of GBM_TREE_COUNTS through staged prediction (section 7). The configuration
    with the lowest mean over the folds is chosen, a tie going to the first in
    the order of candidate_settings, and refitted on the whole sample to
    predict the test rows. The test rows play no part in the choice. Its cost
    is every fit from the first fold to the refit, with the seconds of each
    fit call.
    """
    folds = cv_folds(pair_train, year, split_id, n, draw)
    candidates = candidate_settings()
    fold_deviances = {setting["config"]: [] for setting in candidates}
    seconds, fits = 0.0, 0
    for fold in range(CV_FOLDS):
        validation = folds == fold
        X_fit, y_fit = X_train[~validation], y_train[~validation]
        X_validation, y_validation = X_train[validation], y_train[validation]
        where = (year, split_id, n, draw, fold)
        for setting in candidates:
            if setting["model"] == "gbm":
                continue
            fitted = fit_setting(setting, X_fit, y_fit, X_validation, seed)
            seconds += fitted["seconds"]
            fits += 1
            fold_deviances[setting["config"]].append(validation_deviance(y_validation, fitted["predicted"], where))
        for depth in GBM_DEPTHS:
            start = time.perf_counter()
            boosted = make_gbm(depth, seed).fit(X_fit, y_fit)
            seconds += time.perf_counter() - start
            fits += 1
            # One fit with GBM_MAX_TREES trees gives the predictions after every count in GBM_TREE_COUNTS.
            for trees, predicted in enumerate(boosted.staged_predict(X_validation), start=1):
                if trees in GBM_TREE_COUNTS:
                    deviance = validation_deviance(y_validation, predicted, where)
                    fold_deviances[f"gbm_depth{depth}_trees{trees}"].append(deviance)
    # A configuration replaces the one chosen so far only if its mean is
    # strictly lower, so a tie stays with the earlier one in the order of candidates.
    chosen = candidates[0]
    for setting in candidates[1:]:
        if np.mean(fold_deviances[setting["config"]]) < np.mean(fold_deviances[chosen["config"]]):
            chosen = setting
    refit = fit_setting(chosen, X_train, y_train, X_test, seed)
    fits += 1
    assert fits == CV_SELECT_FITS, f"cv_select made {fits} fits, not CV_SELECT_FITS {CV_SELECT_FITS}."
    return {"setting": chosen, "predicted": refit["predicted"], "seconds": seconds + refit["seconds"], "fits": fits}


def run_analysis_draw(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draw, settings, cv_draws):
    """Fit the methods of stage 5 on one training sample of the analysis year and predict the test rows.

    The sample is drawn with rng("draw", year, split_id, n, draw) and every
    model gets seed_int("model", year, split_id, n, draw), as in run_draw.
    ppml, rf and gbm are fitted once each at the settings frozen for this size
    (settings, from frozen_settings), the ridge line with its own
    cross-validation (fit_ridge), and, on the draws below cv_draws, cv_select.
    The test rows play no part in any fit or choice.
    """
    positions = draw_sample(len(y_pool), year, split_id, n, draw)
    X_train, y_train, pair_train = X_pool[positions], y_pool[positions], pair_pool[positions]
    seed = seed_int("model", year, split_id, n, draw)
    fitted = {model: fit_setting(settings[model], X_train, y_train, X_test, seed) for model in CANDIDATES}
    ridge = fit_ridge(X_train, y_train, pair_train, X_test, year, split_id, n, draw)
    selected = None
    if draw < cv_draws:
        selected = cv_select(X_train, y_train, pair_train, X_test, year, split_id, n, draw, seed)
    return {"fitted": fitted, "ridge": ridge, "cv_select": selected, "largest_train_flow": y_train.max()}


def run_analysis_draws(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draws, settings, cv_draws):
    """Run the draws 0 to draws - 1 of one size of stage 5 and yield their results in the order of the draws.

    As run_draws: one after the other with N_JOBS = 1, otherwise spread over
    N_JOBS worker processes started with "forkserver". Each draw is keyed by
    its own numbers, so nothing depends on the number of workers.
    """
    if N_JOBS == 1:
        for draw in range(draws):
            yield run_analysis_draw(X_pool, y_pool, pair_pool, X_test, y_test, year, split_id, n, draw, settings,
                                    cv_draws)
        return
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS, mp_context=context) as executor:
        yield from executor.map(run_analysis_draw, repeat(X_pool), repeat(y_pool), repeat(pair_pool), repeat(X_test),
                                repeat(y_test), repeat(year), repeat(split_id), repeat(n), range(draws),
                                repeat(settings), repeat(cv_draws))


def score_row(keys, method, setting, predicted, y_test, seconds, fits):
    """One row of analysis_runs.csv: the method, the model and setting behind it, its scores, seconds and fits."""
    assert np.isfinite(predicted).all() and (predicted > 0).all(), \
        f"Predictions that are not positive and finite at {keys}, {method}. Stop and report."
    row = dict(keys)
    row["method"] = method
    row.update(setting)
    mean_test_flow = y_test.mean()
    row["test_deviance"] = mean_poisson_deviance(y_test, predicted)
    row["d2"] = d2_tweedie_score(y_test, predicted, power=1)
    row["mae_rel"] = mean_absolute_error(y_test, predicted) / mean_test_flow
    row["rmse_rel"] = root_mean_squared_error(y_test, predicted) / mean_test_flow
    row["seconds"] = seconds
    row["fits"] = fits
    return row


def analysis_rows(result, y_test, settings, keys):
    """The rows of one draw for analysis_runs.csv, one per method (stage 5 steps 1 to 5).

    keys holds the columns that name the draw: year, split, size and draw.
    size_rule is a copy of the row of the model the size rule chooses at this
    size, with its one fit. best_of_three is a copy of the row of whichever of
    ppml, rf and gbm has the lowest test deviance on this draw (a tie goes to
    the first of the three); it looks at the test set, so it is a reference,
    and its cost is the three fits. cv_select is there only on the draws that
    ran it.
    """
    rows = {}
    for model in CANDIDATES:
        fitted = result["fitted"][model]
        rows[model] = score_row(keys, model, settings[model], fitted["predicted"], y_test, fitted["seconds"], 1)
    # The copies are made before the ppml row gets its own columns below.
    rows["size_rule"] = dict(rows[settings["chosen"]])
    rows["size_rule"]["method"] = "size_rule"
    lowest = CANDIDATES[0]
    for model in CANDIDATES[1:]:
        # Strictly lower only, so that a tie stays with the earlier model.
        if rows[model]["test_deviance"] < rows[lowest]["test_deviance"]:
            lowest = model
    rows["best_of_three"] = dict(rows[lowest])
    rows["best_of_three"]["method"] = "best_of_three"
    rows["best_of_three"]["fits"] = len(CANDIDATES)
    rows["best_of_three"]["seconds"] = sum(rows[model]["seconds"] for model in CANDIDATES)
    selected = result["cv_select"]
    if selected is not None:
        rows["cv_select"] = score_row(keys, "cv_select", selected["setting"], selected["predicted"], y_test,
                                      selected["seconds"], selected["fits"])
    ridge = result["ridge"]
    rows["ridge"] = score_row(keys, "ridge", {"model": "ridge", "config": "ridge"}, ridge["test_predicted"], y_test,
                              ridge["seconds"], RIDGE_FITS)
    rows["ridge"]["ridge_alpha"] = ridge["alpha"]
    # What stage 3 records about the ppml fit (section 8), on the ppml row only.
    ppml = result["fitted"]["ppml"]["ppml"]
    rows["ppml"]["constant_columns"] = ppml["constant_columns"]
    rows["ppml"]["lbfgs_fallback"] = int(ppml["fell_back"])
    rows["ppml"]["train_total_ratio"] = ppml["total_ratio"]
    return [rows[method] for method in METHODS if method in rows]


def analysis_runs_table(rows):
    """The rows of analysis_rows as a table, with the columns of analysis_runs.csv in order."""
    table = pd.DataFrame(rows).reindex(columns=ANALYSIS_RUN_COLUMNS)
    for column in ANALYSIS_WHOLE_NUMBER_COLUMNS:
        table[column] = table[column].astype("Int64")
    return table


def method_draws(runs):
    """One row per size and draw, one column per method, holding its test deviance on that draw (empty if not run)."""
    wide = runs.pivot(index=["size", "draw"], columns="method", values="test_deviance").reset_index()
    wide.columns.name = None
    return wide


def the_only(values):
    """The value, if every entry of values is the same one; otherwise an empty text."""
    distinct = values.unique()
    return distinct[0] if len(distinct) == 1 else ""


def analysis_summary(runs):
    """analysis_summary.csv: one row per size and method (stage 5).

    Every row has the method's mean, standard error and median test deviance
    over its draws, se_unreliable (one draw holds more than half of the sum),
    and its mean seconds and fits per draw; model and config where they are the
    same on every draw. best is the better of rf and gbm at that size by mean
    test deviance (a tie goes to rf). Then, each on one method's rows:
    - ppml: the draws in which ppml has the lower test deviance than rf, gbm and
      best on the same training sample (decision 12; a tie is not a win);
    - size_rule: the mean paired difference size_rule minus cv_select over the
      draws that have both, its standard error, and within_2se, whether the
      difference is at most twice its standard error (the test of H2); and the
      mean paired difference size_rule minus best_of_three with its standard error;
    - ridge: the mean paired difference ridge minus ppml with its standard
      error, and the draws in which ridge beats best (decisions 20 and 22).
    """
    deviance = method_draws(runs)
    rows = []
    for size, at_size in runs.groupby("size"):
        wide = deviance[deviance["size"] == size]
        best = "rf" if wide["rf"].mean() <= wide["gbm"].mean() else "gbm"
        for method in METHODS:
            of_method = at_size[at_size["method"] == method]
            values = of_method["test_deviance"]
            row = {"size": size, "method": method, "model": the_only(of_method["model"]),
                   "config": the_only(of_method["config"]), "draws": len(values), "mean": values.mean(),
                   "se": values.std() / np.sqrt(len(values)), "se_unreliable": values.max() / values.sum() > 0.5,
                   "median": values.median(), "seconds": of_method["seconds"].mean(),
                   "fits": of_method["fits"].mean()}
            if method == "ppml":
                row["best"] = best
                row["ppml_beats_rf"] = int((wide["ppml"] < wide["rf"]).sum())
                row["ppml_beats_gbm"] = int((wide["ppml"] < wide["gbm"]).sum())
                row["ppml_beats_best"] = int((wide["ppml"] < wide[best]).sum())
            if method == "size_rule":
                both = wide[wide["cv_select"].notna()]
                difference = both["size_rule"] - both["cv_select"]
                row["minus_cv_select"] = difference.mean()
                row["se_minus_cv_select"] = paired_se(difference)
                row["draws_with_cv_select"] = len(difference)
                row["within_2se"] = row["minus_cv_select"] <= 2 * row["se_minus_cv_select"]
                difference = wide["size_rule"] - wide["best_of_three"]
                row["minus_best_of_three"] = difference.mean()
                row["se_minus_best_of_three"] = paired_se(difference)
            if method == "ridge":
                difference = wide["ridge"] - wide["ppml"]
                row["minus_ppml"] = difference.mean()
                row["se_minus_ppml"] = paired_se(difference)
                row["ridge_beats_best"] = int((wide["ridge"] < wide[best]).sum())
            rows.append(row)
    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    # fits is the same on every draw of a method, so its mean is a whole number.
    for column in ["fits", "ppml_beats_rf", "ppml_beats_gbm", "ppml_beats_best", "draws_with_cv_select",
                   "ridge_beats_best"]:
        table[column] = table[column].astype("Int64")
    # True or False on the size_rule rows, empty on the others.
    table["within_2se"] = table["within_2se"].astype("boolean")
    return table


def analysis_matrices(runs):
    """At every size, the test deviance of ppml, rf, gbm and ridge on every draw: one row per draw, in order."""
    matrices = {}
    for n in sorted(int(size) for size in runs["size"].unique()):
        at_size = runs[(runs["size"] == n) & runs["method"].isin(CUTOFF_MODELS)]
        wide = at_size.pivot(index="draw", columns="method", values="test_deviance").sort_index()
        matrix = wide[list(CUTOFF_MODELS)].to_numpy()
        assert np.isfinite(matrix).all(), f"A model is missing a draw at size {n}."
        matrices[n] = matrix
    return matrices


def analysis_gaps(matrices, positions):
    """The gaps of analysis_cutoff.csv at every size, on one set of draws.

    positions maps each size to the rows of its matrix to use: a resample of the
    draws, or every draw once. At each size the mean test deviance of each
    model is taken over those rows, best is the better of rf and gbm (a tie
    goes to rf), and each comparison's gap is the mean of its model minus that
    of best. Nothing is tuned: the settings are the frozen ones.
    """
    rows = []
    for n, matrix in matrices.items():
        means = dict(zip(CUTOFF_MODELS, matrix[positions[n]].mean(axis=0)))
        best = "rf" if means["rf"] <= means["gbm"] else "gbm"
        row = {"size": n, "best": best}
        for comparison, model in CUTOFF_COMPARISONS.items():
            row[f"gap_{comparison}"] = means[model] - means[best]
        rows.append(row)
    return pd.DataFrame(rows)


def check_analysis_gaps(matrices, summary):
    """Before the bootstrap: on every draw once, analysis_gaps must give best and the gaps of the means in summary.

    The gaps are compared to a relative 1e-9 of the means: the two add up the
    same draws in a different order, so they can differ in the last digits.
    """
    every_draw = {n: np.arange(len(matrix)) for n, matrix in matrices.items()}
    again = analysis_gaps(matrices, every_draw)
    for row in again.itertuples():
        means = summary[summary["size"] == row.size].set_index("method")["mean"]
        best = summary.loc[(summary["size"] == row.size) & (summary["method"] == "ppml"), "best"].item()
        assert row.best == best, f"The bootstrap's better ensemble differs from the summary at size {row.size}."
        for comparison, model in CUTOFF_COMPARISONS.items():
            expected = means[model] - means[best]
            scale = max(abs(means[model]), abs(means[best]))
            assert abs(getattr(row, f"gap_{comparison}") - expected) <= 1e-9 * scale, \
                f"The bootstrap's gap differs from the summary at size {row.size}, {comparison}."
    return again


def reversal_sizes(sizes, gap, pattern):
    """The two sizes around the change from zero or positive to negative, on reversal and reversal_then_crossing.

    find_crossing gives no n_star for these patterns; decision 12 reports the
    reversal below last_crossing, so stage 5 says where it is. Empty on the
    other patterns.
    """
    result = {"reversal_from": np.nan, "reversal_to": np.nan}
    if pattern not in ("reversal", "reversal_then_crossing"):
        return result
    sizes = np.asarray(sizes)
    negative = np.asarray(gap, dtype=float) < 0
    for i in range(len(negative) - 1):
        if not negative[i] and negative[i + 1]:
            result["reversal_from"], result["reversal_to"] = int(sizes[i]), int(sizes[i + 1])
            return result
    return result


def analysis_crossings(gap_table):
    """find_crossing on the gap of each comparison of analysis_cutoff.csv, with the sizes around a reversal."""
    rows = []
    for comparison in CUTOFF_COMPARISONS:
        gap = gap_table[f"gap_{comparison}"]
        result = find_crossing(gap_table["size"], gap)
        rows.append({"comparison": comparison, "sizes": len(gap_table), **result,
                     **reversal_sizes(gap_table["size"], gap, result["pattern"])})
    return pd.DataFrame(rows)


def bootstrap_analysis(matrices, year, split_id, resamples):
    """find_crossing on resampled draws, for each comparison of analysis_cutoff.csv.

    As in stage 4 step 6: in resample b, at each size n separately, the draws
    are drawn again with replacement, as many as there are, with
    rng("bootstrap", year, split_id, n, b), the same drawn draws for every
    model and for both comparisons. Nothing is tuned inside a resample.
    Returns one row per resample and comparison.
    """
    rows = []
    for resample in range(resamples):
        positions = {}
        for n, matrix in matrices.items():
            draws = len(matrix)
            positions[n] = rng("bootstrap", year, split_id, n, resample).integers(0, draws, size=draws)
        resampled = analysis_gaps(matrices, positions)
        for comparison in CUTOFF_COMPARISONS:
            result = find_crossing(resampled["size"], resampled[f"gap_{comparison}"])
            rows.append({"resample": resample, "comparison": comparison, "pattern": result["pattern"],
                         "n_star": result["n_star"], "last_crossing": result["last_crossing"]})
    return pd.DataFrame(rows)


def analysis_cutoff_table(crossing_table, resampled):
    """analysis_cutoff.csv: the columns of cutoff.csv (cutoff_table), and the sizes around a reversal."""
    table = cutoff_table(crossing_table, resampled)
    for column in ["reversal_from", "reversal_to"]:
        table[column] = table[column].astype("Int64")
    return table


def selector_choices(runs, selector):
    """On the draws that have both selectors: the model the selector picked, the winner, and the difference.

    The winner is the model best_of_three found best on the draw (a tie goes to
    the first of ppml, rf, gbm). For cv_select the model picked is the model of
    its chosen configuration, whatever its setting. The difference is the
    selector's test deviance minus that of best_of_three; for cv_select it can
    be negative, since its configuration need not be one of the three frozen
    settings. The result is what selection_confusion reads.
    """
    deviance = method_draws(runs)
    picked = runs[runs["method"] == selector][["size", "draw", "model"]].rename(columns={"model": "picked"})
    choices = deviance[deviance["cv_select"].notna()].merge(picked, on=["size", "draw"])
    choices["winner"] = winners(choices)
    choices["difference"] = choices[selector] - choices["best_of_three"]
    return choices[["size", "draw", "picked", "winner", "difference"]]


def selection_confusion_table(runs):
    """selection_confusion.csv: for size_rule and cv_select, the model picked against the model that won.

    For each selector and each size, and for all sizes together (sizes =
    all): one row per model the selector picked, with the draws in which each
    of ppml, rf and gbm won (selection_confusion; at one size the size rule
    picks one model, so it has one such row there). Then a row picked = total
    with every draw of that scope, the share picked right, the share that
    always picking the most frequent winner of each size would get, and the
    mean difference to best_of_three with its paired standard error.
    """
    tables = []
    for selector in ("size_rule", "cv_select"):
        counts, scores = selection_confusion(selector_choices(runs, selector))
        rows = []
        for score in scores.to_dict("records"):
            in_scope = counts[counts["sizes"] == score["sizes"]]
            rows.extend(in_scope.to_dict("records"))
            total = {"sizes": score["sizes"], "picked": "total", "draws": score["draws"]}
            for winner in CANDIDATES:
                total[f"won_{winner}"] = int(in_scope[f"won_{winner}"].sum())
            total.update({key: score[key] for key in ["share_picked_right", "share_most_frequent_winner",
                                                       "mean_difference", "se_difference"]})
            rows.append(total)
        table = pd.DataFrame(rows)
        table.insert(0, "selector", selector)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------------
# Stage 6: the three checks on the development year (section 6)
# ---------------------------------------------------------------------------

# The three checks, in the order stage 6 runs them, as they appear in the
# column variant of its files. The base experiment of stages 3 and 4 is "base".
CHECKS = ("noise", "split2", "trees2x")


def experiment_cutoff(runs, components, year, split_id, resamples):
    """Steps 1, 2 and 4 to 6 of stage 4 on the files of one experiment.

    The same functions as stage 4, in the same order: the summary by size and
    configuration, rf and gbm tuned on this experiment's own test set, the
    gaps of ppml with its win counts, the pattern of each comparison, the
    check that the resampling code gives back the tuned settings and the
    gaps, and the bootstrap with rng("bootstrap", year, split_id, n, b).
    Returns the tuned settings, the table of gaps (the columns of gaps.csv)
    and the cutoff (the columns of cutoff.csv).
    """
    summary = summarise(runs)
    tuned = tune(summary)
    wins = ppml_wins(per_draw(runs, line_configs(tuned)), better_ensemble(tuned))
    gap_table = gaps_with_wins(gaps(components, runs, tuned), wins)
    crossing_table = crossings(gap_table)
    matrices = candidate_matrices(runs)
    check_resampling(matrices, tuned, gap_table)
    resampled = bootstrap_crossings(matrices, year, split_id, resamples)
    return {"tuned": tuned, "gaps": gap_table, "cutoff": cutoff_table(crossing_table, resampled)}


def trees2x_counts(tuned_trees):
    """The tree counts at which the check trees2x scores gbm: the tuned count, twice it, and the new maximum."""
    return sorted({tuned_trees, 2 * tuned_trees, 2 * GBM_MAX_TREES})


def run_trees2x_draw(X_pool, y_pool, X_test, year, split_id, n, draw, rf_setting, gbm_setting):
    """The check trees2x on one training sample: rf with twice the trees, gbm with twice the largest number of trees.

    The sample and the seed are those of the base experiment,
    rng("draw", year, split_id, n, draw) and seed_int("model", year, split_id,
    n, draw). rf is fitted at its tuned setting (rf_setting) with 2 * RF_TREES
    trees, without out-of-bag predictions, and scored with all its trees and
    with its first RF_TREES trees alone. scikit-learn draws the seed of each
    tree in turn from the one seed, so these first trees are the trees of the
    base forest, and with them alone the forest of the base run comes back.
    gbm is fitted once at its tuned depth (gbm_setting) with 2 * GBM_MAX_TREES
    trees and scored, through staged prediction, at the counts of
    trees2x_counts, the first of which is its tuned count: again the model of
    the base run. Returns, for each configuration, its setting, its test
    predictions and the seconds of the fit it came from, with the largest
    training flow.
    """
    positions = draw_sample(len(y_pool), year, split_id, n, draw)
    X_train, y_train = X_pool[positions], y_pool[positions]
    seed = seed_int("model", year, split_id, n, draw)
    entries = []

    forest = make_rf(rf_max_features(rf_setting["rf_features"]), rf_setting["rf_min_leaf"], seed, oob=False,
                     trees=2 * RF_TREES)
    start = time.perf_counter()
    forest.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    # The forest of the base run: the mean of the first RF_TREES trees, added up
    # in the order scikit-learn's predict adds them.
    first_trees = np.zeros(len(X_test))
    for tree in forest.estimators_[:RF_TREES]:
        first_trees += tree.predict(X_test)
    first_trees /= RF_TREES
    rf_settings = {"model": "rf", "rf_features": rf_setting["rf_features"], "rf_min_leaf": rf_setting["rf_min_leaf"]}
    entries.append({"setting": {**rf_settings, "config": rf_setting["config"]}, "predicted": first_trees,
                    "seconds": seconds})
    entries.append({"setting": {**rf_settings, "config": f"{rf_setting['config']}_trees{2 * RF_TREES}"},
                    "predicted": forest.predict(X_test), "seconds": seconds})

    depth = gbm_setting["gbm_depth"]
    counts = trees2x_counts(gbm_setting["gbm_trees"])
    boosted = make_gbm(depth, seed, trees=2 * GBM_MAX_TREES)
    start = time.perf_counter()
    boosted.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    for trees, predicted in enumerate(boosted.staged_predict(X_test), start=1):
        if trees in counts:
            setting = {"model": "gbm", "config": f"gbm_depth{depth}_trees{trees}", "gbm_depth": depth,
                       "gbm_trees": trees}
            entries.append({"setting": setting, "predicted": predicted, "seconds": seconds})
    return {"entries": entries, "largest_train_flow": y_train.max()}


def run_trees2x_draws(X_pool, y_pool, X_test, year, split_id, n, draws, rf_setting, gbm_setting):
    """Run the draws 0 to draws - 1 of the check trees2x and yield their results in the order of the draws.

    As run_draws: one after the other with N_JOBS = 1, otherwise spread over
    N_JOBS worker processes started with "forkserver".
    """
    if N_JOBS == 1:
        for draw in range(draws):
            yield run_trees2x_draw(X_pool, y_pool, X_test, year, split_id, n, draw, rf_setting, gbm_setting)
        return
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS, mp_context=context) as executor:
        yield from executor.map(run_trees2x_draw, repeat(X_pool), repeat(y_pool), repeat(X_test), repeat(year),
                                repeat(split_id), repeat(n), range(draws), repeat(rf_setting), repeat(gbm_setting))


def trees2x_rows(result, y_test, groups, keys):
    """The rows of one draw of the check trees2x, with the columns of dev_runs.csv.

    The scores are those of stage 3 (test_scores) with the largest prediction
    divided by the largest training flow and the seconds of the fit. There is
    no training deviance: the forest is fitted without out-of-bag predictions.
    """
    rows = []
    for entry in result["entries"]:
        predicted = entry["predicted"]
        where = f"{keys}, {entry['setting']['config']}"
        assert np.isfinite(predicted).all() and (predicted > 0).all(), \
            f"Predictions that are not positive and finite at {where}. Stop and report."
        row = dict(keys)
        row.update(entry["setting"])
        row.update(test_scores(y_test, predicted, groups, where))
        row["max_pred_ratio"] = predicted.max() / result["largest_train_flow"]
        row["seconds"] = entry["seconds"]
        rows.append(row)
    return rows


def trees2x_differences(check_runs, base_runs, tuned):
    """For each configuration of the check trees2x: its test deviance against the tuned one of the base run.

    tuned holds the tuned rf and gbm of the base experiment at the size of the
    check (rows of size_rule.csv). On every draw, the difference is the
    check's test deviance minus that of the tuned configuration of the same
    model on the same training sample in base_runs (dev_runs.csv). Returns, per
    configuration, the mean of both, the mean difference, its paired standard
    error, and the largest difference on one draw relative to the base value:
    for the configurations that are the tuned ones themselves, this checks
    that the check redid the base run.
    """
    rows = []
    for config, of_config in check_runs.groupby("config", sort=False):
        model = of_config["model"].iloc[0]
        size = of_config["size"].iloc[0]
        base_config = tuned.loc[(tuned["size"] == size) & (tuned["model"] == model), "config"].item()
        base = base_runs[(base_runs["size"] == size) & (base_runs["config"] == base_config)]
        merged = of_config.merge(base[["draw", "test_deviance"]], on="draw", suffixes=("", "_base"))
        assert len(merged) == len(of_config), f"Some draw of {config} has no base row of {base_config}."
        difference = merged["test_deviance"] - merged["test_deviance_base"]
        rows.append({"size": size, "model": model, "config": config, "base_config": base_config,
                     "draws": len(merged), "mean": merged["test_deviance"].mean(),
                     "base_mean": merged["test_deviance_base"].mean(), "difference": difference.mean(),
                     "se_difference": paired_se(difference),
                     "largest_relative_difference": float((difference.abs() / merged["test_deviance_base"]).max())})
    return pd.DataFrame(rows)
