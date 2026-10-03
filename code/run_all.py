# Drafted with Claude (Anthropic). Asked for: a short script that runs the
# stage notebooks in order, each one in a fresh kernel.
"""Run the project from start to finish (Requirements 4.7 and 4.8).

From the repository folder:

    python code/run_all.py              stages 2 to 4, using the data in data/
    python code/run_all.py --download   stage 1 first, then stages 2 to 4

Each notebook is executed top to bottom in a newly started kernel. The run
stops at the first error. The notebooks themselves are not modified: what a
run produces is whatever the notebooks save in data/ and figures/.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parent

DOWNLOAD_STAGE = "01_get_data.ipynb"
STAGES = ["02_clean.ipynb", "03_model.ipynb", "04_figures.ipynb"]


def run_notebook(name, scratch):
    """Execute one notebook in a new kernel and stop if any cell fails."""
    print(f"Running {name}", flush=True)
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
    )


def main():
    stages = list(STAGES)
    if "--download" in sys.argv[1:]:
        stages.insert(0, DOWNLOAD_STAGE)
    with tempfile.TemporaryDirectory() as scratch:
        for name in stages:
            run_notebook(name, scratch)
    print("All stages finished.")


if __name__ == "__main__":
    main()
