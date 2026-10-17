# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UGR'16 v2 network attack detection: PCA baseline vs MLP, LSTM and 1D-CNN autoencoders,
scored by reconstruction error. `docs/DSA4266.md` is the final proposal and the source of
truth for method decisions; `docs/eda-findings.md` holds the data evidence behind it.

Week 1 starter files are in place: each pipeline module in `src/ugr16ad/` has its
interface, a `NotImplementedError` body naming its owner, and tests marked `@todo`
(strict xfail, from `tests/stubs.py`). Ownership and the open decisions are in
`docs/team-plan.md`. Do not implement a member's stub unless asked; the stubs are
their own work (`models/pca.py` is Chen Lei's week-2 stub, the rest are week 1).
When a stub is implemented and its tests pass, they report XPASS and fail; delete the
`@todo` marker on those tests.

## Commands

The project is developed on Windows. `make` is unavailable, so invoke tools directly.

```bash
# Setup (the venv is not committed; create it if missing)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e .

# Tests (all, one file, one test)
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m pytest tests/test_windows.py
./.venv/Scripts/python.exe -m pytest tests/test_windows.py::test_name

# Lint (CI runs both; `format --check` fails CI, plain `format` rewrites)
./.venv/Scripts/python.exe -m ruff check src tests
./.venv/Scripts/python.exe -m ruff format src tests

# EDA checks (read-only, ~1 min)
./.venv/Scripts/python.exe scripts/eda_checks.py
```

CI (`.github/workflows/ci.yml`) runs on Ubuntu, Python 3.11, with the CPU-only torch
wheel, on pushes to `main` and `working` and on PRs.

## Architecture

Every model goes through one common pipeline, so a difference in results comes from
the model, not the plumbing (load, preprocess, window, train/score, threshold, save):

1. `data.load_dataset` reads the CSVs and returns four chronological `Part`s keyed
   `fit`, `tune`, `calib`, `test`; each carries a `segment` id that changes at every
   data gap.
2. `preprocess.Preprocessor` is fit on `fit` only, then transforms every part.
3. `windows.eligible_positions` / `make_windows` build windows that never cross a
   segment boundary. All models, including single-minute ones, are scored on the same
   eligible positions.
4. `train.train_autoencoder` / `score_autoencoder` are shared by all neural models; a
   model owner only writes the `nn.Module`, which must return the same shape as its
   input. `models/pca.py` returns scores in the same `(scores, feature_errors)` form.
5. `evaluate` fits the frozen threshold on `calib` and computes threshold metrics
   (what an analyst would see) separately from ranking metrics (retrospective only).
6. `results.save_run` writes `reports/results/<model>/seed<seed>/` (`scores.csv`,
   `meta.json`, optional `.npy`). Charts and tables read only these files, never
   model code.

`synthetic.inject_anomalies` is the validation sanity check, since validation has no
labelled attacks.

## Data

`data/external/ugr16v2/UGR16v2.{X,Y}{train,test}.csv` is committed. Resolve it through
`ugr16ad.paths.UGR16_DIR`. `Row` (`YYYYMMDDHHMM`, local time) is the join key between X
and Y.

## Rules the proposal fixes (do not relitigate without the team)

- **One-class framing.** The training file has zero positive minutes in all seven attack
  labels, so models fit background traffic only. Validation measures reconstruction, not
  detection. Detection is measured on test only.
- **Target:** a minute is an attack if any of the seven non-blacklist label counts is > 0.
  `labelblacklist` is in neither X nor y.
- **Never a model input:** `Row`, or anything derived from it (hour, weekday, day index).
  Test attacks follow an injection schedule that calendar features would learn.
- **Chronological splits only.** Train rows 0–78,608 fit models; rows 78,609–98,261 are
  validation, split into an earlier tuning block and a later calibration block. No
  shuffling, no random folds.
- **Fit on the model-fitting rows only:** log1p + standard scaler, constant-feature
  removal, PCA. Apply unchanged to validation and test.
- **Threshold:** 99th percentile of calibration-block scores, then frozen. Test-derived
  operating points (recall at FPR ≤ 1%, partial AUC) are reported as retrospective
  comparisons, never used to replace it.
- **Windows** end at minute t, contain only t and earlier minutes, and never cross a data
  gap (DST hour on 2016-03-27, outage 2016-04-01 → 04-06) or the train/test boundary.
  The detection score is the final minute's reconstruction MSE.
- Test-label inspection is dataset characterisation only. It must not set window length,
  thresholds or any other model setting.

## Conventions

- Docstrings explain **why**, not what. Match that when adding code.
- Tests use tiny synthetic fixtures; only the path tests touch the real CSVs.
- `data/raw`, `data/interim`, `data/processed` and `reports/*` are gitignored;
  `data/external/` is committed.
- The earlier vulnerability-prioritisation project is archived at git tag
  `archive/vulnprio`. Nothing from it applies here.
