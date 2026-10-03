# Drafted with Claude (Anthropic). Asked for: folder locations that work on
# any computer, so that no notebook needs an absolute path, and one random
# seed shared by every notebook.
"""Code used by more than one notebook (Requirement 4.9).

In a notebook:

    from helpers import DATA, FIGURES, SEED
"""
from pathlib import Path

# The repository folder, wherever it has been cloned.
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

# One seed for every random step, so the numbers in the paper can be reproduced.
SEED = 2026
