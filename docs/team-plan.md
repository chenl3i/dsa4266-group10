# Team plan: 12 Oct – 5 Nov 2026

Basically in week 1 each of us builds one part of the common pipeline that every
model will use (data loading, input prep, scoring, training tools), then from
week 2 each of us owns one model.

## Who owns which files

| Person | Week 1: common pipeline | Tests that define "done" | Week 2–3: model and analysis |
|---|---|---|---|
| Phone | `src/ugr16ad/data.py`, `src/ugr16ad/synthetic.py` | `tests/test_data.py`, `tests/test_synthetic.py` | LSTM autoencoder (`src/ugr16ad/models/lstm.py`), GRU if time |
| Kesha | `src/ugr16ad/preprocess.py`, `src/ugr16ad/windows.py` | `tests/test_preprocess.py`, `tests/test_windows.py` | 1D-CNN autoencoder (`src/ugr16ad/models/cnn.py`) |
| Vanessa | `src/ugr16ad/evaluate.py`, `src/ugr16ad/results.py` | `tests/test_evaluate.py`, `tests/test_results.py` | Charts: score over time, loss curves, which features cause alerts, latent space; synthetic-anomaly check on every model |
| Chen Lei | `src/ugr16ad/train.py`, `src/ugr16ad/models/pca.py` | `tests/test_train.py`, `tests/test_pca.py` | PCA tuning, MLP autoencoder (`src/ugr16ad/models/mlp.py`), MLP on flattened windows |

Week 2 files don't exist yet; each owner creates theirs.

## How the pieces connect

```
data.load_dataset()        -> {"fit", "tune", "calib", "test"} Parts      (Phone)
Preprocessor.fit(fit.X)    -> transform every part                         (Kesha)
make_windows(...)          -> windows for LSTM / CNN / flattened MLP       (Kesha)
train_autoencoder(...)     -> trained model + loss curve                   (Chen Lei)
score_autoencoder / PCAScorer.score -> (scores, feature_errors)            (Chen Lei)
fit_threshold(calib scores), threshold_metrics, ranking_metrics            (Vanessa)
save_run(...)              -> reports/results/<model>/seed<seed>/          (Vanessa)
```

Every function's inputs and outputs are written in its docstring. Code against
those, not against someone's unfinished implementation, and nobody waits.

## How to do your week 1 part

1. `git checkout main && git pull`, then `git checkout -b <yourname>/week1`.
2. Set up once: `python -m venv .venv`, then `pip install -r requirements.txt` and `pip install -e .`.
3. Open your files. Replace each `raise NotImplementedError(...)` with real code.
4. Run your tests: `python -m pytest tests/test_<yourfile>.py -v`.
   - `XFAIL` = still a stub. `FAILED` with an assertion = your code disagrees with the test.
   - `FAILED ... XPASS(strict)` = it works. Delete the `@todo` line above that test.
5. Done when your tests pass with no `@todo` left, and `ruff check src tests` plus `ruff format src tests` are clean.
6. Push your branch and open a pull request into `main`. CI must be green.

If you think a test or a signature is wrong, say so in the group chat before
changing it. Someone else is coding against it.

## Open decisions (settle on the Mon 19 Oct call)

- **Tune / calibrate split of the validation rows.** `data.TUNE_FRACTION` is a 0.5 placeholder.
- **Window length.** The proposal says it is fixed before evaluation, from operational reasoning and prior literature (Phone + Kesha to propose).
- **Synthetic anomaly size.** `rate`, `magnitude`, `n_features` for the sanity check (Phone to propose).

## Schedule

| When | What |
|---|---|
| 12–18 Oct | Week 1: common pipeline (table above) |
| Mon 19 Oct | Call: merge week 1, settle open decisions |
| 19–25 Oct | Week 2: first version of each model; charts built on PCA |
| Mon 26 Oct | Call: first results; slides start |
| 26 Oct–1 Nov | Week 3: fix settings, run 3 seeds; report sections |
| Mon 2 Nov | Call: results review |
| Tue 3 Nov | Results lock |
| Wed 4 Nov | Timed rehearsal (10 min + Q&A) |
| Thu 5 Nov | Presentation and report |

Daily: one line in the group chat (done / next / blocked).
Whatever you build, write its report paragraph the same week.
