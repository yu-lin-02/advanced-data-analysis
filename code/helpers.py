# Drafted with Claude (Anthropic). Asked for: folder locations that work on
# any computer, so that no notebook needs an absolute path, and one random
# seed shared by every notebook. Later asked for: the folder of the raw
# download and the two years of the project.
"""Code used by more than one notebook (Requirement 4.9).

In a notebook:

    from helpers import DATA, FIGURES, SEED
"""
from pathlib import Path

# The repository folder, wherever it has been cloned.
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
# The CEPII download: too large for GitHub, so .gitignore leaves it out.
RAW = DATA / "raw"
FIGURES = ROOT / "figures"

# One seed for every random step, so the numbers in the paper can be reproduced.
SEED = 2026

# The development year (for exploring and tuning) and the analysis year (for
# confirming), chosen by Yu on 6 October 2026.
DEV_YEAR = 2017
ANALYSIS_YEAR = 2019
