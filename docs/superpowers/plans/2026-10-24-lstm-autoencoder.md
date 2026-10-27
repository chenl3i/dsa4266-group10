# LSTM Autoencoder (Week 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Never add Claude co-author trailers or "Generated with Claude Code" lines to commits or PRs.

## Context

Phone's week 2 deliverable (`docs/team-plan.md`) is the LSTM autoencoder, with GRU "if time".
The common pipeline is merged into `working`; Kesha (#6 CNN) and Vanessa (#5 windowed MLP)
have open PRs. The Mon 26 Oct call expects first results, so this plan delivers the model
**and** a run script that writes a standard results folder. Window length is agreed at 20.

**Goal:** `LSTMAutoencoder` (LSTM or GRU cell) plus `scripts/run_lstm.py` producing
`reports/results/<lstm|gru>/seed<N>/` via the shared pipeline.

**Architecture:** Repeat-vector seq2seq: encoder RNN → final hidden state → Linear to a
`latent_dim` bottleneck → repeated over the window → decoder RNN → Linear back to features.
Training and scoring reuse `train.train_autoencoder` / `score_autoencoder` unchanged; the run
script only wires existing pipeline functions together.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, pytest, ruff.

**Spec:** `docs/DSA4266.md` (§ Recurrent autoencoder, windows paragraph) and `CLAUDE.md`
(rules the proposal fixes). Branch: `phone/week2` (currently equal to `origin/working`).

## Global Constraints

- `nn.Module` takes float32 `(batch, length, features)` and returns the **same shape** (`train.py` docstring).
- Score = final minute's MSE, via `score_autoencoder` only; never a custom scoring rule.
- Never a model input: `Row` or anything derived from it.
- Fit `Preprocessor` on `fit` only; train on `fit` windows, early-stop on `tune`, threshold on `calib` (`fit_threshold`, 0.99), frozen for `test`.
- Windows from `windows.make_windows` only (never cross a segment). Window length default `20`.
- Call `train.set_seed(seed)` **before** building the model.
- Docstrings explain why, not what. Tests use tiny synthetic fixtures, never the real CSVs.
- Ruff: line length 100; `ruff check src tests scripts` and `ruff format --check src tests scripts` clean.
- Python: `./.venv/Scripts/python.exe`.

## Review Focus

1. Window shorter than any segment in a part → zero windows; run script must raise a clear `ValueError` naming the part, not crash inside torch. Test in Task 2.
2. `cell` other than `"lstm"`/`"gru"` → `ValueError` at construction. Test in Task 1.
3. Output must not depend on batch composition (eval mode, no batch-norm) → scoring the same window alone or in a batch gives the same score. Test in Task 1.
4. Score timestamps must be each window's **final** minute (`part.X.index[positions]`), not its first. Test in Task 2.
5. `latents.npy` / `feature_errors.npy` rows align with `scores.csv` rows across all four parts (concatenated in the same order). Test in Task 2.

---

### Task 1: `LSTMAutoencoder` model

**Files:**
- Create: `src/ugr16ad/models/lstm.py`
- Test: `tests/test_lstm.py`

**Interfaces:**
- Produces: `class LSTMAutoencoder(nn.Module)` with
  `__init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 16, cell: str = "lstm")`,
  `encode(self, x: Tensor) -> Tensor` (`(B, W, F)` → `(B, latent_dim)`),
  `forward(self, x: Tensor) -> Tensor` (`(B, W, F)` → `(B, W, F)`). Attributes `n_features`, `cell`.
  No `window_length` arg: the decoder repeats the latent `x.shape[1]` times, so any length works (say so in the class docstring).

- [ ] **Step 1: Write failing tests** in `tests/test_lstm.py` (fixtures: `W, F = 5, 6`; `make_model(seed=0, cell="lstm")` calls `set_seed(seed)` then builds `hidden_dim=16, latent_dim=4`; `low_rank_windows(n=128)` as in Vanessa's PR: rank-3 data reshaped to `(n, W, F)` float32):
  - `test_output_shape_matches_input_shape` (parametrize `cell` over `["lstm", "gru"]`, and `W` over `[1, 5, 20]`): `model(x).shape == x.shape`.
  - `test_encode_returns_the_bottleneck`: `encode(torch.randn(8, W, F)).shape == (8, 4)`.
  - `test_gru_cell_uses_gru`: `any(isinstance(m, nn.GRU) for m in make_model(cell="gru").modules())` and no `nn.LSTM`.
  - `test_rejects_bad_arguments`: `ValueError` for `n_features=0`, `hidden_dim=0`, `latent_dim=0`, `cell="rnn"`.
  - `test_training_lowers_the_loss`: `train_autoencoder(make_model(), d[:96], d[96:], epochs=40, batch_size=16, lr=1e-2, patience=100)`; last `train_loss` < first.
  - `test_scores_use_the_final_minute_only`: after 3 epochs, `score_autoencoder` feature errors equal `(recon[:, -1] - d[:, -1])**2` (rtol 1e-5, atol 1e-6).
  - `test_score_does_not_depend_on_batch`: scores of `d` with `batch_size=1` equal those with `batch_size=1024` (atol 1e-6).
  - `test_same_seed_reproduces_the_scores`: two runs with `make_model(seed=1)` and `seed=1` give identical scores.
- [ ] **Step 2: Run, expect ImportError/FAIL:** `./.venv/Scripts/python.exe -m pytest tests/test_lstm.py -v`
- [ ] **Step 3: Implement** `src/ugr16ad/models/lstm.py`. Module docstring: why a recurrent AE (tests whether temporal dependencies improve detection, per proposal). Use `batch_first=True`, single layer; `rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]`. Final hidden: for LSTM `h_n` is `output[1][0]`, for GRU `output[1]`; take `[-1]`.
- [ ] **Step 4: Run, expect all PASS**, plus `ruff check` / `ruff format --check` on the two files.
- [ ] **Step 5: Commit** `git add src/ugr16ad/models/lstm.py tests/test_lstm.py && git commit -m "Add LSTM/GRU autoencoder"`

### Task 2: Run script

**Files:**
- Create: `scripts/run_lstm.py`
- Test: `tests/test_run_lstm.py` (load the script with `importlib.util.spec_from_file_location`, since `scripts/` is not a package)

**Interfaces:**
- Consumes: `LSTMAutoencoder` (Task 1); `data.load_dataset`, `data.Part`; `preprocess.Preprocessor`; `windows.make_windows`; `train.set_seed/train_autoencoder/score_autoencoder`; `evaluate.fit_threshold/threshold_metrics/ranking_metrics/per_category_recall`; `results.run_dir/save_run`.
- Produces: `run(parts: dict[str, Part], *, seed: int, out_root: Path, window_length: int = 20, cell: str = "lstm", hidden_dim: int = 64, latent_dim: int = 16, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3, patience: int = 5, device: str = "cpu") -> Path` (returns the run directory) and `main()` with argparse flags of the same names (`--seed` repeatable/list, default `[0]`), calling `run(load_dataset(), ...)` with `out_root=RESULTS_DIR`.

- [ ] **Step 1: Write failing tests.** Fixture `tiny_parts()`: four `Part`s built directly, 3 feature columns of nonnegative counts on a minute `DatetimeIndex`, `labels` with the 7 `data.ATTACK_LABELS` columns, `y = labels.sum(axis=1) > 0`; `fit` 60 rows with segments `[0]*30 + [1]*30`, `tune`/`calib`/`test` 30 rows each, single segment; `test` has a few labelled minutes. Call `run(parts, seed=0, out_root=tmp_path, window_length=5, epochs=2, hidden_dim=8, latent_dim=2)`.
  - `test_writes_a_loadable_run`: `load_run(dir)` works; `meta["model"] == "lstm"`, `meta["seed"] == 0`, `meta["window_length"] == 5`; dir == `tmp_path / "lstm" / "seed0"`.
  - `test_threshold_is_the_calib_quantile`: `meta["threshold"] == fit_threshold(scores[scores.part == "calib"].score)`.
  - `test_scores_cover_only_eligible_final_minutes`: per part, rows == `len(eligible_positions(segment, 5))` (fit: 2×26 = 52; others 26), and timestamps equal `part.X.index[positions]`.
  - `test_arrays_align_with_scores`: `latents.shape == (len(scores), 2)`, `feature_errors.shape == (len(scores), 3)`; parts appear in order fit, tune, calib, test.
  - `test_gru_writes_under_gru`: `cell="gru"` → dir `tmp_path / "gru" / "seed0"`.
  - `test_too_long_window_raises`: `window_length=40` → `ValueError` whose message names `"fit"` (the first part checked; both fit segments are 30 rows).
- [ ] **Step 2: Run, expect FAIL:** `./.venv/Scripts/python.exe -m pytest tests/test_run_lstm.py -v`
- [ ] **Step 3: Implement `run`.** Order: `Preprocessor().fit(parts["fit"].X)`; transform each part; `make_windows(X, part.segment.to_numpy(), window_length)` per part, raise `ValueError(f"no {window_length}-minute windows in {name}")` if empty; `set_seed(seed)`; build model; `train_autoencoder(model, w["fit"], w["tune"], ...)`; `score_autoencoder` each part; latents via `model.encode` in `torch.no_grad()` batches on `device`, returned `.cpu().numpy()`; threshold from calib; test metrics = `threshold_metrics` + `ranking_metrics` on `part.y.to_numpy()[positions]`, `per_category_recall(test.labels.iloc[positions], alerts)` stored as records. `meta`: `model` (= cell), `seed`, `threshold`, `window_length`, `hidden_dim`, `latent_dim`, `lr`, `batch_size`, `best_epoch`, `epochs_run`, `train_seconds`, `n_params`, `test_metrics`, `per_category_recall`. Also write `history.csv` (the loss curve) into the run dir. Log with `logging_utils.get_logger`.
- [ ] **Step 4: Run, expect PASS;** full suite `./.venv/Scripts/python.exe -m pytest` still green; ruff clean incl. `scripts`.
- [ ] **Step 5: Commit** `git add scripts/run_lstm.py tests/test_run_lstm.py && git commit -m "Add LSTM run script writing the standard results format"`

### Task 3: Real-data run and PR (done by Phone / main session, not a subagent)

- [ ] **Step 1:** `./.venv/Scripts/python.exe scripts/run_lstm.py --seed 0` → expect `reports/results/lstm/seed0/{scores.csv,meta.json,latents.npy,feature_errors.npy,history.csv}`; check `alert_rate` on calib ≈ 1%, `best_epoch` < `epochs`, note `train_seconds`.
- [ ] **Step 2:** If time: `--cell gru --seed 0` the same way.
- [ ] **Step 3:** Push `phone/week2`, open PR into `working` titled "Week 2: LSTM autoencoder" with the seed-0 test metrics in the body.

## Verification

- `./.venv/Scripts/python.exe -m pytest` — all pass (pca still 7 xfail).
- `./.venv/Scripts/python.exe -m ruff check src tests scripts` and `ruff format --check src tests scripts` — clean.
- Task 3 Step 1 output folder loads with `results.load_run`.

## Notes for Phone (not tasks)

- `CLAUDE.md` has an uncommitted edit from this session; commit or stash it before starting.
- Window length 20 is still listed as open in `docs/team-plan.md`; record it there and consider one shared constant with Kesha/Vanessa.
- Synthetic sanity check is skipped until `rate`/`magnitude`/`n_features` are agreed.
