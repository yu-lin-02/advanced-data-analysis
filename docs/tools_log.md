# Tools log

<!-- Drafted with Claude (Anthropic). Asked for (18 October 2026, section 5 of docs/build_plan.md): a log with one line per session in which a tool worked on this project, from which Yu writes the appendix of the paper (Requirements 5.2 and 6.6). -->

`CLAUDE.md` and `docs/build_plan.md` were drafted with Claude in a claude.ai chat on 3 October 2026, asked for as instructions for Claude Code, and the plan was revised with Claude on 6, 14, 17 and 18 October 2026 (the 17 October revision added the checkpoint and the freeze, and the 18 October revision added decisions 13 to 16).

Each line: the date, the tool, what was asked, the files touched.

## Earlier work

Drafted by Claude from the project record on 18 October 2026. TODO: Yu confirms or corrects these lines.

- 1 to 3 October 2026. Claude (Anthropic), in the claude.ai project. Asked for: the steps to set up GitHub, Nuvolos and Overleaf, a repository skeleton and a workflow that reruns the project on every push, the package versions of the Nuvolos workspace, and an empty frame of the paper in the SIAM template. Files: `README.md`, `.gitignore`, `requirements.txt`, `code/helpers.py`, `code/run_all.py`, the four empty stage notebooks, `data/README.md`, `figures/README.md`, `.github/workflows/ci.yml` (pull requests #1 and #2); the paper frame in Overleaf.
- 2 to 4 October 2026. Claude, in the claude.ai project. Asked for: comments on Yu's notes on the project's theme and on candidate data sets, a draft of the paper's sections, a read-only check of the CEPII file on Yu's computer, the 200-word proposal, and `CLAUDE.md` and `docs/build_plan.md`. Files: none in the repository; the paper draft is kept outside it.
- 6 October 2026. Claude, in the claude.ai project. Asked for: the choice of estimator, target and score (decision 8), a revision of the plan, the proposal rewritten from Yu's text, and stage 1. Files: `code/01_get_data.ipynb`, `code/helpers.py`, `data/gravity_extract.csv`, `data/README.md`, `.gitignore` (pull request #3).
- 9 October 2026. Claude. Asked for: answers to the teaching assistant's questions on the method. Files: none.
- 14 October 2026. Claude. Asked for: a check of nonlinear least squares against PPML, which Yu then dropped, and the open decisions of the plan. Files: none.
- 16 October 2026. Claude. Asked for: an explanation of the Poisson model. Files: none.
- 17 October 2026. Claude. Asked for: a pause in stage 3 to review intermediate results (the checkpoint), and why PPML trails the ensembles at the smallest sizes in a preview run on 2017. Files: none in the repository; the preview's numbers are not used.
- 18 October 2026. Claude. Asked for: answers to Yu's note of that day, the history of the colonial-tie indicator, and the decisions left before implementation. Files: none in the repository.

## Sessions

- 18 October 2026. Claude, in the claude.ai project. Asked for: Phase A of the build plan, the plumbing of section 5 and stage 2. Files: `CLAUDE.md`, `docs/build_plan.md`, `docs/tools_log.md`, `code/helpers.py`, `code/run_all.py`, `code/01_get_data.ipynb` (the import of `helpers` moved to the top), `code/02_clean.ipynb`, `code/03_development.ipynb` and `code/07_figures.ipynb` (renamed placeholders), `code/04_size_rule.ipynb`, `code/05_analysis.ipynb` and `code/06_checks.ipynb` (new placeholders), `.github/workflows/ci.yml`, `.gitignore`, `README.md`, `data/README.md`, `figures/README.md`, `results/README.md`.
