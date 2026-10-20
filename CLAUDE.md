<!-- Drafted with Claude (Anthropic). Asked for: an instruction file that Claude Code reads at the start of every session in this repository, holding the course rules that bind the code and the way of working. The build plan itself is in docs/build_plan.md. Place this file in the repository folder, next to README.md. Added to the repository on 18 October 2026 with three changes: Yu's name in place of pronouns, the command for part 1 of stage 3, and the checkpoint notebook in the rule on stages. -->

# Instructions for Claude Code

This repository is Yu Lin's individual semester project for Advanced Data Analytics (HEC Lausanne, autumn 2026). It is public, it is graded, and it is the only assessment of the course. Yu is the author and has to be able to explain everything in it. You write the code Yu asks for. You do not make Yu's decisions.

## Read first

- `docs/build_plan.md` holds the design, the facts checked on the data, and the specification of every stage. At the start of every session read its Status block (section 0) and its Decisions table (section 2), then the section of the stage you are asked to work on. Do not work from what you remember of an earlier session.
- The paper draft is kept outside this repository. If Yu gives you text that disagrees with the plan, stop and ask Yu which one holds.

## The gate (Requirement 3.4)

"A project may not be undertaken before the sign-off has been received."

- While the Status block says the sign-off has not been received, work only on Phase A: the plumbing of section 5, stage 1 (get the data) and stage 2 (clean it and look at it).
- If Yu asks for work on stages 3 to 7 while the block says no, quote the requirement and ask Yu to update the block first. Only Yu edits the Status block.
- If the sign-off changed the scope, revise the plan with Yu before writing code.

## Rules from the course requirements

- One notebook per stage, numbered, in `code/`. Each runs from top to bottom in a freshly started kernel, starts from the file the previous stage saved and ends by saving what the next stage needs (4.7, 4.8). No file holds the whole analysis. The one exception to the order is `code/03b_checkpoint.ipynb`, run once between part 1 and the full run of stage 3 (section 6 of the plan).
- Code used by more than one notebook lives in `code/helpers.py` and is imported, never copied (4.9).
- Every figure and table for the paper is written by `code/07_figures.ipynb` from files in `results/` and `data/` (4.11). Nothing in `figures/` is made or edited by hand.
- Use only the packages listed in `requirements.txt`, at the pinned versions, and the standard library (4.10). To add a package, ask Yu: the version has to be the one installed in the course's Nuvolos workspace.
- No caches, checkpoints, virtual environments or editor settings in the repository (4.12). No absolute paths: use `ROOT`, `DATA`, `RESULTS` and `FIGURES` from `helpers.py`. No keys or passwords.
- The project has to run in the Nuvolos workspace (4.22). The README states the running time measured there (4.24).
- Never rewrite history, force-push, delete the repository or make it private (4.20).

## Honest results (Requirement 7.3)

"Data or results adjusted to produce an effect are academic misconduct."

- Design choices are fixed in the plan before results exist. Never change a setting, a sample, a seed, a year or a cleaning step because of how a result looks. If a change seems needed, stop, say why, and let Yu decide. Yu's decision goes into the Decisions table with the reason and the date.
- If a check in the plan fails, or a count differs from the plan, stop and report both numbers. Do not edit the data, the check or the expected value to make it pass.
- Never type a number into the README, a notebook's text, a table or the paper that code in this repository did not produce. Never describe a result that has not been run. A hypothesis that fails is a result: report it as it is.
- The files committed in `results/` and `figures/` come from a full run in the Nuvolos workspace. Output of quick mode, or of a run on another machine, is never committed and never quoted.

## Attribution (Requirements 5.2, 6.1, 6.6)

- Every file you create or change carries a note in the style already used here: `Drafted with Claude (Anthropic). Asked for: <what Yu asked for>.` It is a comment at the top of a `.py`, `.yml` or `.txt` file, an HTML comment in a Markdown file, and an italic line in the first cell of a notebook.
- At the end of each session add one line to `docs/tools_log.md`: the date, the tool, what was asked, the files touched. Yu writes the appendix of the paper from this log.
- Do not paste code from websites, repositories, papers or other students. If you adapt something from library documentation, say so in a comment with the link (6.4). Code whose origin you cannot state does not go into this repository (6.7).

## How to write the code

- Yu has to explain every line (5.4). Write plain Python: small functions, clear names, comments that say why. No classes, no decorators, no clever one-liners.
- Each notebook opens with a Markdown cell: the stage, what it reads, what it writes, the attribution line. Each step has a short Markdown cell saying what it does, and prints the row counts or values the paper will quote.
- Every random step takes its generator from `helpers.rng(...)`, which is built from the one `SEED`. No unseeded random call anywhere.
- When a stage is finished, write a walk-through for Yu in the pull request, in plain words: what each cell and each helper does, which choices it contains, and which of them Yu should be ready to defend.

## Commands

- Install: `pip install -r requirements.txt`
- Everything, from the committed extract: `python code/run_all.py`. Add `--download` to run stage 1 first.
- One stage: `python code/run_all.py --only 03_development.ipynb`
- Part 1 of stage 3, the sizes up to `CHECKPOINT_MAX`, before the checkpoint: `python code/run_all.py --part 1`
- Quick test of the plumbing: `ADA_QUICK=1 python code/run_all.py`. Few sizes and draws, output in `results_quick/` and `figures_quick/`, which git ignores.
- Workers for the long stages: `ADA_JOBS=4 python code/run_all.py`. The results must not depend on the number of workers.

## Git

- Never commit to `main`. One branch and one pull request per stage, with small commits whose messages say what changed and why.
- The pull request states what Yu asked for, what was done, which checks ran and what they printed.
- The `reproduce` workflow in `.github/workflows/ci.yml` has to pass before you ask Yu to merge.
