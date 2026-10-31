# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UGR'16 v2 network attack detection: PCA baseline vs MLP, LSTM and 1D-CNN autoencoders,
scored by reconstruction error. `docs/DSA4266.md` is the final proposal and the source of
truth for method decisions; `docs/eda-findings.md` holds the data evidence behind it.

## Status and ownership

Weeks 1 and 2 (the common pipeline and all five models) are merged into `working`.
Week 3 is experiments and report sections; ownership and schedule are in
`docs/team-plan.md`. Do not write a member's model unless asked; it is their own work.

- Models and their run entry points: `pca.py` and `mlp.py` (Chen Lei,
  `python -m ugr16ad.experiments`), `lstm.py` (Phone, `scripts/run_lstm.py`), `cnn.py`
  (Kesha, `scripts/run_cnn.py`), `windowed_mlp.py` (Vanessa,
  `scripts/run_windowed_mlp.py`). `scripts/make_comparison_table.py` reads them all.
- Shared names: model constructors take `n_features` first; `meta.json` uses
  `n_parameters`, `train_seconds`, `inference_seconds` (calib + test scoring),
  `calib_alert_rate`, `window_length`, `latent_dim`, `test_metrics` and
  `per_category_recall`. The comparison table reads these keys, so keep them.
- Settled on 26 Oct: window length 20, `data.TUNE_FRACTION` 0.5, synthetic sanity
  check `rate=0.01` with `(magnitude, n_features)` in `(2, 1), (2, 5), (4, 1), (4, 5)`
  on preprocessed calib (`synthetic.SANITY_GRID`).
- Still open, for the team to settle, not for Claude to pick: the comparison table
  columns.

Branches: `<name>/weekN` → PR into `working`; `main` is updated from `working`.

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
./.venv/Scripts/python.exe -m pytest -m "not slow"   # skip the real-CSV checks

# Model runs (write reports/results/<model>/seed<seed>/; see scripts/README.md)
./.venv/Scripts/python.exe -m ugr16ad.experiments tune --output <dir>
./.venv/Scripts/python.exe -m ugr16ad.experiments evaluate --output <dir> --window 20
./.venv/Scripts/python.exe scripts/run_lstm.py --seed 0 1 2        # --cell gru for GRU
./.venv/Scripts/python.exe scripts/make_comparison_table.py

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
   eligible positions. `make_windows` materialises a full copy (`m × length ×
   features`), so cast to float32 before windowing the fit part.
4. `train.train_autoencoder` / `score_autoencoder` are shared by all neural models; a
   model owner only writes the `nn.Module`. It takes float32 `(batch, features)` or
   `(batch, length, features)` and must return the same shape (a CNN transposes to
   channels-first internally and back). Training loss is MSE over the whole window,
   with early stopping on `tune` loss and best weights restored; scores use only the
   final minute. Call `train.set_seed` before building the model. `models/pca.py`
   returns the same `(scores, feature_errors)` form.
5. `evaluate` fits the frozen threshold on `calib` and computes threshold metrics
   (what an analyst would see) separately from ranking metrics (retrospective only).
6. `results.save_run` writes `reports/results/<model>/seed<seed>/`: `scores.csv`
   (`timestamp, part, score`; a window's timestamp is its final minute), `meta.json`
   (at least `model`, `seed`, `threshold`), optional `latents.npy` /
   `feature_errors.npy`. Charts and tables read only these files, never model code.

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
- Tests use tiny synthetic fixtures. Only the path tests and the `@pytest.mark.slow`
  tests (real-data checks in `test_data`, `test_preprocess`, `test_windows`) touch the
  real CSVs.
- `data/raw`, `data/interim`, `data/processed` and `reports/*` are gitignored;
  `data/external/` is committed.
- The earlier vulnerability-prioritisation project is archived at git tag
  `archive/vulnprio`. Nothing from it applies here.
