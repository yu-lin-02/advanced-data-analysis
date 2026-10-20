# Drafted with Claude (Anthropic). Asked for: a short script that runs the
# stage notebooks in order, each one in a fresh kernel. Later asked for (18
# October 2026, section 5 of docs/build_plan.md): the seven stages, running
# one stage alone, part 1 of stage 3 before the checkpoint, one thread per
# model, the minutes each stage took, and what each notebook prints.
"""Run the project from start to finish (Requirements 4.7 and 4.8).

From the repository folder:

    python code/run_all.py                           stages 2 to 7, from the data in data/
    python code/run_all.py --download                stage 1 first, then stages 2 to 7
    python code/run_all.py --only 02_clean.ipynb     one stage only
    python code/run_all.py --part 1                  part 1 of stage 3 only (sizes up to CHECKPOINT_MAX);
                                                     the checkpoint is run afterwards, on its own

    ADA_QUICK=1 python code/run_all.py               quick test that the code runs
    ADA_JOBS=4 python code/run_all.py                four workers for the long stages

Each notebook is executed top to bottom in a newly started kernel. The run
stops at the first error. The notebooks themselves are not modified: what a
run produces is whatever the notebooks save in data/, results/ and figures/.
What each notebook prints is shown here once the notebook has finished.

03b_checkpoint.ipynb is not in the order of a full run: it is run once, at
the checkpoint between part 1 and the full run of stage 3 (section 6 of the
build plan). In quick mode it runs right after stage 3, so that every push
tests it.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# helpers comes before pandas, as in every notebook (section 7 of the build plan).
from helpers import CHECKPOINT_MAX, QUICK, RESULTS

import pandas as pd

CODE = Path(__file__).resolve().parent

DOWNLOAD_STAGE = "01_get_data.ipynb"
STAGES = [
    "02_clean.ipynb",
    "03_development.ipynb",
    "04_size_rule.ipynb",
    "05_analysis.ipynb",
    "06_checks.ipynb",
    "07_figures.ipynb",
]
DEVELOPMENT_STAGE = "03_development.ipynb"
CHECKPOINT_STAGE = "03b_checkpoint.ipynb"
# The two files stage 3 writes, which the checkpoint reads.
DEVELOPMENT_FILES = ["dev_runs.csv", "dev_components.csv"]

USAGE = """Usage:
    python code/run_all.py [--download]
    python code/run_all.py --only <notebook>
    python code/run_all.py --part 1"""


def kernel_environment(part):
    """The environment of each kernel: one thread per model, and ADA_PART for part 1."""
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    environment.pop("ADA_PART", None)
    if part is not None:
        environment["ADA_PART"] = part
    return environment


def run_notebook(name, scratch, environment):
    """Execute one notebook in a new kernel, stop if any cell fails, return the minutes it took."""
    print(f"Running {name}", flush=True)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.kernel_name=python3",
            "--ExecutePreprocessor.timeout=-1",
            "--output-dir", scratch,
            str(CODE / name),
        ],
        check=True,
        env=environment,
    )
    print_outputs(Path(scratch) / name)
    return (time.perf_counter() - start) / 60


def print_outputs(executed):
    """Print what the executed notebook printed: nbconvert itself shows none of it."""
    notebook = json.loads(executed.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output["output_type"] == "stream":
                print("".join(output["text"]), end="", flush=True)


def copy_checkpoint_inputs():
    """Quick mode: copy the rows of the sizes up to CHECKPOINT_MAX into the checkpoint's inputs.

    Returns False, after saying why, when stage 3 or the checkpoint notebook
    does not exist yet, so that the checkpoint is skipped.
    """
    sources = [RESULTS / name for name in DEVELOPMENT_FILES]
    if not (CODE / CHECKPOINT_STAGE).exists() or not all(source.exists() for source in sources):
        print(f"Skipping {CHECKPOINT_STAGE}: it runs once stage 3 writes "
              f"{' and '.join(DEVELOPMENT_FILES)} and the notebook exists.", flush=True)
        return False
    inputs = RESULTS / "checkpoint" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for source in sources:
        # Stage 3 keys every row by year, split, size and draw; "size" is the training size.
        table = pd.read_csv(source)
        table[table["size"] <= CHECKPOINT_MAX].to_csv(inputs / source.name, index=False, lineterminator="\n")
    print(f"Copied the rows of the sizes up to {CHECKPOINT_MAX} into {inputs.relative_to(CODE.parent)}", flush=True)
    return True


def read_arguments(arguments):
    """Return the notebooks to run and the value of ADA_PART, or stop with the usage."""
    if not arguments:
        return list(STAGES), None
    if arguments == ["--download"]:
        return [DOWNLOAD_STAGE] + STAGES, None
    if len(arguments) == 2 and arguments[0] == "--only":
        name = arguments[1]
        if not (CODE / name).exists():
            sys.exit(f"No notebook {name} in code/.\n{USAGE}")
        return [name], None
    if arguments == ["--part", "1"]:
        return [DEVELOPMENT_STAGE], "1"
    sys.exit(f"Unknown or combined options: {' '.join(arguments)}\n"
             f"--part 1 cannot be combined with --only or --download.\n{USAGE}")


def main():
    stages, part = read_arguments(sys.argv[1:])
    # run_all.py sets ADA_PART itself, so that part 1 is always asked for in the same way.
    if part is None and "ADA_PART" in os.environ:
        sys.exit("ADA_PART is set in this shell, but run_all.py sets it itself. Run part 1 with: "
                 "python code/run_all.py --part 1. To run every size, unset ADA_PART first.")
    environment = kernel_environment(part)
    if part is not None:
        print(f"Part {part} of stage 3 only: the sizes up to {CHECKPOINT_MAX}. No other stage runs; "
              "the checkpoint is run afterwards (section 6 of the build plan).")
    minutes = []
    with tempfile.TemporaryDirectory() as scratch:
        for name in stages:
            minutes.append((name, run_notebook(name, scratch, environment)))
            # In quick mode the checkpoint follows stage 3 in the order of a full run.
            if QUICK and name == DEVELOPMENT_STAGE and len(stages) > 1 and copy_checkpoint_inputs():
                minutes.append((CHECKPOINT_STAGE, run_notebook(CHECKPOINT_STAGE, scratch, environment)))
    print("Minutes per stage:")
    for name, value in minutes:
        print(f"  {name:<24} {value:6.1f}")
    print(f"  {'total':<24} {sum(value for _, value in minutes):6.1f}")
    print("All stages finished.")


if __name__ == "__main__":
    main()
