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
# checkpoint report that stages 4, 5 and 7 will reuse.
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
    return configurations


CONFIGURATIONS = model_configurations()
# The three groups of test rows whose deviance stage 3 records separately.
GROUP_NAMES = ("top1", "next9", "rest")
# The columns of dev_runs.csv, in order (stage 3 step 5). The coefficient and
# ones_ columns are filled on the ppml rows only (decision 14).
RUN_COLUMNS = (
    ["variant", "year", "split", "size", "draw", "model", "config",
     "rf_features", "rf_min_leaf", "gbm_depth", "gbm_trees",
     "test_deviance", "d2", "mae_rel", "rmse_rel", "train_deviance"]
    + [f"deviance_{name}" for name in GROUP_NAMES]
    + ["max_pred_ratio", "seconds", "constant_columns", "lbfgs_fallback", "train_total_ratio", "coef_intercept"]
    + [f"coef_{name}" for name in PREDICTORS]
    + [f"ones_{name}" for name in INDICATORS]
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
    is scored on the same rows. Rows are ranked by flow, largest first, and
    the counts are rounded down; equal flows keep their order in the test set.
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


def make_ppml():
    """The gravity equation estimated by PPML. alpha=0: the default alpha=1 would add a ridge penalty."""
    return PoissonRegressor(alpha=0, solver="newton-cholesky")


def make_rf(max_features, min_leaf, seed):
    """A random forest with the Poisson loss, fitted on one thread.

    oob_score=True keeps the out-of-bag predictions for the training deviance.
    It leaves the predictions unchanged.
    """
    return RandomForestRegressor(n_estimators=RF_TREES, criterion="poisson", max_features=max_features,
                                 min_samples_leaf=min_leaf, oob_score=True, n_jobs=1, random_state=seed)


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


def run_draw(X_pool, y_pool, X_test, y_test, year, split_id, n, draw):
    """Fit the 13 models on one training sample and predict the test rows for every configuration (section 7).

    The sample is drawn with rng("draw", year, split_id, n, draw) and every
    model gets the same random_state, seed_int("model", year, split_id, n,
    draw), so a draw gives the same numbers whichever worker runs it. Returns,
    in the order of CONFIGURATIONS, the test predictions, the mean Poisson
    deviance on the training sample (out-of-bag for rf, none for tree) and the
    seconds of the fit each configuration came from (the fit alone, as stage 3
    step 5 defines them: predicting and scoring are not timed); and what
    stage 3 records about the sample and the ppml fit.
    """
    # Decision 15: the ridge line is written in its own pull request, only if Yu adds it at the checkpoint.
    assert not RIDGE_LINE, "RIDGE_LINE is True, but the ridge line has not been written yet (decision 15)."
    assert max(GBM_TREE_COUNTS) <= GBM_MAX_TREES, "A tree count of GBM_TREE_COUNTS is above GBM_MAX_TREES."
    positions = draw_sample(len(y_pool), year, split_id, n, draw)
    X_train, y_train = X_pool[positions], y_pool[positions]
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

    order = [configuration["config"] for configuration in CONFIGURATIONS]
    return {
        "test_predicted": np.array([test_predicted[config] for config in order]),
        "train_deviance": np.array([train_deviance[config] for config in order]),
        "seconds": np.array([seconds[config] for config in order]),
        "largest_train_flow": y_train.max(),
        "constant_columns": ppml["constant_columns"],
        "ppml": ppml,
    }


def run_draws(X_pool, y_pool, X_test, y_test, year, split_id, n, draws):
    """Run the draws 0 to draws - 1 of one size and yield their results in the order of the draws.

    With N_JOBS = 1 the draws run one after the other in this process. With
    more, they are spread over N_JOBS worker processes, started with
    "forkserver" (section 7): a worker forked from a process that has already
    fitted a boosted model can hang. Each draw is keyed by its own numbers, and
    the results come back in the order of the draws, so nothing depends on the
    number of workers.
    """
    if N_JOBS == 1:
        for draw in range(draws):
            yield run_draw(X_pool, y_pool, X_test, y_test, year, split_id, n, draw)
        return
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS, mp_context=context) as executor:
        yield from executor.map(run_draw, repeat(X_pool), repeat(y_pool), repeat(X_test), repeat(y_test),
                                repeat(year), repeat(split_id), repeat(n), range(draws))


def check_predictions(predicted, keys):
    """Stop unless every prediction is positive and finite (section 8): stage 3 takes their logs."""
    good = np.isfinite(predicted).all(axis=1) & (predicted > 0).all(axis=1)
    bad = [configuration["config"] for configuration, ok in zip(CONFIGURATIONS, good) if not ok]
    assert not bad, f"Predictions that are not positive and finite at {keys}: {bad}. Stop and report."


def draw_rows(result, y_test, groups, keys):
    """The rows of one draw for dev_runs.csv, one per configuration (stage 3 step 5).

    keys holds the columns that name the draw: variant, year, split, size and draw.
    """
    rows = []
    mean_test_flow = y_test.mean()
    for i, configuration in enumerate(CONFIGURATIONS):
        predicted = result["test_predicted"][i]
        row = dict(keys)
        row.update(configuration)
        row["test_deviance"] = mean_poisson_deviance(y_test, predicted)
        row["d2"] = d2_tweedie_score(y_test, predicted, power=1)
        row["mae_rel"] = mean_absolute_error(y_test, predicted) / mean_test_flow
        row["rmse_rel"] = root_mean_squared_error(y_test, predicted) / mean_test_flow
        row["train_deviance"] = result["train_deviance"][i]
        # Each group's deviance is summed over its rows and divided by all the
        # test rows, so the three add up to the test deviance.
        unit = unit_deviance(y_test, predicted)
        for group, name in enumerate(GROUP_NAMES):
            row[f"deviance_{name}"] = unit[groups == group].sum() / len(y_test)
        groups_total = sum(row[f"deviance_{name}"] for name in GROUP_NAMES)
        assert np.isclose(groups_total, row["test_deviance"], rtol=1e-8, atol=0), \
            f"The three groups do not add up to the test deviance at {keys}, {configuration['config']}."
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
# candidates (rf and gbm at their tuned setting) and the two reference lines.
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


# Grid edges (section 6, Checkpoint): for each setting, its grid in order, and
# the step beyond the first and the last value, fixed before part 1. None
# means the grid cannot be extended on that side.
EDGE_GRIDS = {
    ("rf", "rf_features"): {"grid": sorted(rf_features(value) for value in RF_MAX_FEATURES),
                            "below": "2 predictors per split", "above": None},
    ("rf", "rf_min_leaf"): {"grid": sorted(RF_MIN_LEAF), "below": None, "above": "min_samples_leaf 50"},
    ("gbm", "gbm_depth"): {"grid": sorted(GBM_DEPTHS), "below": None, "above": "depth 7"},
    ("gbm", "gbm_trees"): {"grid": sorted(GBM_TREE_COUNTS), "below": "10 trees (staged, no extra fit)",
                           "above": "750 and 1,000 trees, with GBM_MAX_TREES 1,000"},
}


def edges(runs, summary, tuned):
    """Whether each tuned setting sits at an edge of its grid (checkpoint item 5, rule Grid edges).

    A tuned setting is at an edge when its value is the first or the last of
    that setting's grid, and its mean test deviance is lower than that of the
    configuration with the next value inside the grid, the model's other
    setting held at its tuned value, by more than twice the paired standard
    error. A tie is never an edge.
    """
    rows = []
    for _, chosen in tuned.iterrows():
        for (model, setting), edge in EDGE_GRIDS.items():
            if chosen["model"] != model:
                continue
            grid = edge["grid"]
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
                "extension": (edge[side] or "cannot be extended") if at_edge else "",
            })
    # The columns are named even when no tuned setting is at either end of its grid.
    columns = ["size", "model", "setting", "tuned_value", "side", "neighbour_value", "neighbour_config",
               "neighbour_minus_tuned", "se", "at_edge", "extension"]
    return pd.DataFrame(rows, columns=columns)


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
