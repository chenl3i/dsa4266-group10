# Team plan: 12 Oct – 5 Nov 2026

Basically in week 1 each of us builds one part of the common pipeline that every
model will use (data loading, input prep, scoring, training loop), in week 2 each
of us owns one model, and in week 3 each of us runs our model's experiments and
double-checks one part of the pipeline.

## Who owns which files

| Person | Week 1: common pipeline | Week 2: model | Week 3: final work |
|---|---|---|---|
| Phone | Load CSVs, align features and labels, define targets and chronological splits; synthetic anomalies for the sanity check | LSTM autoencoder (GRU if time) | Run LSTM experiments; check splits and label alignment |
| Kesha | Log transformation, scaling and gap-safe sliding windows | 1D-CNN autoencoder | Run CNN experiments; check preprocessing and window boundaries |
| Vanessa | Threshold calibration, evaluation metrics and standard results format | Windowed MLP autoencoder (flattened windows) | Run windowed MLP experiments; assemble the comparison table |
| Chen Lei | Starter interfaces and shared training loop, including early stopping, seeds and timing | PCA baseline and single-minute MLP autoencoder | Run PCA/MLP experiments; check training settings and reproducibility |

Files and the tests that define "done":

| Person | Week 1 files | Week 1 tests | Week 2 files |
|---|---|---|---|
| Phone | `src/ugr16ad/data.py`, `src/ugr16ad/synthetic.py` | `tests/test_data.py`, `tests/test_synthetic.py` | `src/ugr16ad/models/lstm.py` |
| Kesha | `src/ugr16ad/preprocess.py`, `src/ugr16ad/windows.py` | `tests/test_preprocess.py`, `tests/test_windows.py` | `src/ugr16ad/models/cnn.py` |
| Vanessa | `src/ugr16ad/evaluate.py`, `src/ugr16ad/results.py` | `tests/test_evaluate.py`, `tests/test_results.py` | `src/ugr16ad/models/windowed_mlp.py` |
| Chen Lei | `src/ugr16ad/train.py` | `tests/test_train.py` | `src/ugr16ad/models/pca.py`, `src/ugr16ad/models/mlp.py` |

`models/pca.py` already has its interface and `@todo` tests (`tests/test_pca.py`);
the other week 2 files don't exist yet, so each owner creates theirs.

## How the pieces connect

```
data.load_dataset()        -> {"fit", "tune", "calib", "test"} Parts      (Phone)
Preprocessor.fit(fit.X)    -> transform every part                         (Kesha)
make_windows(...)          -> windows for LSTM / CNN / flattened MLP       (Kesha)
train_autoencoder(...)     -> trained model + loss curve                   (Chen Lei)
score_autoencoder / PCAScorer.score -> (scores, feature_errors)            (Chen Lei; PCA is week 2)
fit_threshold(calib scores), threshold_metrics, ranking_metrics            (Vanessa)
save_run(...)              -> reports/results/<model>/seed<seed>/          (Vanessa)
```

Every function's inputs and outputs are written in its docstring. Code against
those, not against someone's unfinished implementation, and nobody waits.

## How to do your week 1 part

1. `git checkout main && git pull`, then `git checkout -b <yourname>/week1`.
2. Set up once: `python -m venv .venv`, then `./.venv/Scripts/python.exe -m pip install -r requirements.txt` and `./.venv/Scripts/python.exe -m pip install -e .`.
3. Open your files. Replace each `raise NotImplementedError(...)` with real code.
4. Run your tests: `./.venv/Scripts/python.exe -m pytest tests/test_<yourfile>.py -v`.
   - `XFAIL` = still a stub. `FAILED` with an assertion = your code disagrees with the test.
   - `FAILED ... XPASS(strict)` = it works. Delete the `@todo` line above that test.
5. Done when your tests pass with no `@todo` left, and `./.venv/Scripts/python.exe -m ruff check src tests` plus `./.venv/Scripts/python.exe -m ruff format src tests` are clean.
6. Push your branch and open a pull request into `main`. CI must be green.

If you think a test or a signature is wrong, say so in the group chat before
changing it. Someone else is coding against it.

## Phone's week 3 part (26 Oct – 1 Nov)

Settled on the Mon 26 Oct call: window length **20 minutes**, `TUNE_FRACTION` **0.5**
(tune and calib are each ~9,826 minutes, about 6.8 days, so each covers close to a
full weekly cycle).

1. **Fix settings (Tue 27 Oct).** Drop "placeholder" from the `TUNE_FRACTION` comment
   in `data.py`. Synthetic anomaly settings (settled): `rate=0.01`,
   `magnitude ∈ {2, 4}`, `n_features ∈ {1, 5}`, injected into preprocessed calib.
2. **Run LSTM, 3 seeds (Wed 28 Oct).**
   `./.venv/Scripts/python.exe scripts/run_lstm.py --seed 0 1 2 --window-length 20`
   writes `reports/results/lstm/seed{0,1,2}/`. Settings stay at the script defaults;
   nothing is tuned on test.
3. **Synthetic sanity check (Thu 29 Oct).** AUC of LSTM calib scores on the injected
   rows for each setting in the grid. This shows the model reconstructs and reacts;
   it is not a detection result.
4. **Pipeline check: splits and label alignment (Thu–Fri).** Confirm with tests, not by
   eye: fit = rows 0–78,608, validation = 78,609–98,261, tune before calib, no overlap
   or shuffling; segments break at the 2016-03-27 DST hour and the 04-01 → 04-06
   outage; X and Y join on `Row` with no dropped or duplicated minutes; target = any
   of the seven non-blacklist labels > 0; `labelblacklist` absent from X and y.
   Report anything wrong to the owner before fixing it.
5. **GRU (if time).** `--cell gru`, same 3 seeds, saved as its own model.
6. **Report paragraphs (Sat–Sun).** Data and splits section; LSTM method and results
   (mean ± sd over seeds, threshold metrics kept separate from ranking metrics).
7. **Hand results to Vanessa by Sun 1 Nov** for the comparison table.

Blocked: NaN values in `meta.json` need a fix in Vanessa's `save_run`.

## Open decisions

- **Comparison table columns.** Which threshold and ranking metrics, and whether to add training time (Vanessa to propose).

## Schedule

| When | What |
|---|---|
| 12–18 Oct | Week 1: common pipeline (table above) |
| Mon 19 Oct | Call: merge week 1, settle open decisions |
| 19–25 Oct | Week 2: first version of each model |
| Mon 26 Oct | Call: first results; slides start |
| 26 Oct–1 Nov | Week 3: fix settings, run 3 seeds, pipeline checks, comparison table; report sections |
| Mon 2 Nov | Call: results review |
| Tue 3 Nov | Results lock |
| Wed 4 Nov | Timed rehearsal (10 min + Q&A) |
| Thu 5 Nov | Presentation and report |

Daily: one line in the group chat (done / next / blocked).
Whatever you build, write its report paragraph the same week.
