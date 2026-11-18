# Contributing contract

Follow this and branches merge without conversation. Everything below is the rule, not a suggestion.

## 1. Ownership: one area, one owner
Each folder in [OWNERS.md](OWNERS.md) has exactly one owner. **Only edit files inside your own area.**
Need something from another area? Open an issue (label `request`) or send a PR into that area. Never edit it silently.

Shared files (`src/common/*`, `requirements/*`, `pyproject.toml`, `.github/*`, `README.md`) change only by the
rules in section 6.

## 2. Branches
- `main` is always green and always runnable. **No direct pushes** (PR only).
- Branch from latest `main`; name it `<area>/<short-kebab-description>`:

| Area prefix | Folder(s) |
|---|---|
| `data` | `src/data/`, `results/eda/` |
| `feat` | `src/features/` |
| `model` | `src/models/<your files>`, `results/models/` |
| `tune` | `src/tuning/`, `results/tuning/` |
| `hmm` | `src/hmm/`, `results/hmm/` |
| `api` | `src/api/`, `scripts/` |
| `ui` | `ui/` |
| `viz` | `src/viz/`, `results/viz/` |
| `docs` | `docs/`, `README.md` |
| `infra` | `.github/`, `requirements/`, `pyproject.toml` |

Examples: `model/xgboost`, `hmm/viterbi-core`, `feat/arc-heading`, `ui/trip-selector`.
- One branch = one small change (aim for under a day of work). Delete the branch after merge.
- Rebase on `main` before opening a PR: `git fetch && git rebase origin/main`.

## 3. Commits
Conventional commits: `<type>(<area>): <summary>` in the imperative, under 72 chars.
Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `exp` (experiment results).
Examples: `feat(hmm): add log-space viterbi`, `exp(model): xgboost cv results`, `fix(api): handle missing trip`.

## 4. Pull requests
- Target `main`, **squash merge**, title follows the commit format.
- Fill in the PR template. CI (lint + tests) must pass.
- Needs at least one approval from another teammate. Review within a few hours; small PRs make that easy.
- If you only touched your own area and CI is green, you may merge your own PR after approval.

## 5. Plug-in conventions (so nobody edits shared registries)
**Models.** One file per model in `src/models/<name>.py`. It is auto-discovered, nothing else to register. It must define:
```python
NAME = "xgboost"                 # unique, lowercase
FAMILY = "boosting"              # linear | instance | tree | ensemble | boosting | probabilistic | neural | ...
DEFAULT_PARAMS = {...}           # initial hyperparameters (reported in the paper)
SEARCH_SPACE = {...}             # optional: Bayesian search space, {"param": ("float"|"int"|"log", lo, hi)}
def build(**params): ...         # returns an object with fit(X, y) and predict_proba(X) -> (n, 2)
```
Use `src/models/baseline_prior.py` as the reference. `tests/test_contract.py` checks every model file.

**Data contract.** The candidates table schema is in `src/common/schema.py`. Do not change column names or meaning.

**Results.** Write only to your own `results/<area>/` folder, one file per run or model
(e.g. `results/models/xgboost.json`). Never share a results file between two people.

**Randomness.** Use `RANDOM_SEED` from `src/common/schema.py`. Log seeds and configs with results.

## 6. Shared files and contract changes
- `src/common/*` (schema, metrics, paths): change only through a PR labelled `contract-change`. All area owners
  must approve. Prefer adding over modifying; never rename or remove.
- `requirements/*`: one file per concern (`base`, `ml`, `geo`, `api`, `viz`, `dev`). Add your dependency to the
  matching file, one line, alphabetical.
- `README.md`: only the `docs` area edits it. Send your section as a PR into `docs` or an issue.

## 7. Hard rules
1. **Never commit** data (`data/`), PDFs, model binaries over 50 MB, secrets, API keys, `.env` files.
2. **Evaluator trips and their labels are never used** for training, feature selection, model choice or tuning.
   Ground truth stays server-side in the API; it is never sent to the UI or the model.
3. No hardcoded answers for any trip.
4. Fit all learned preprocessing on the 80 training trips only. Split by trip, never by observation.
5. Tune decoder parameters on train + student-test only.
6. Notebooks: clear outputs before committing, keep them in `notebooks/<area>/`, and put reusable code in `src/`.
7. Do not force-push to shared branches. Do not rewrite `main`.
8. Code must be your own team's work. Do not copy from other teams.

## 8. Local setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
ruff check .
```
Put the dataset trip folders in `data/train/` and `data/student_test/` (ignored by git).
Use the GPU for anything torch/CUDA wherever possible.
