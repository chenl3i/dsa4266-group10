# Chen Lei: Week 3 experiment record

## Completed fitting/tuning experiments

Run folder: `reports/results/chenlei-week3-20261026/` (gitignored).
Entry point: `python -m ugr16ad.experiments`.

The run used 78,609 fitting rows, 9,826 tuning rows and 133 retained features.
The remaining 9,827 training-period rows are reserved for calibration. The user confirmed the 50/50 tuning/calibration split (`tune_fraction=0.5`). The user confirmed the shared window length is **20**. This is
an eligibility filter for these single-minute models, not their input length.

Preprocessing was fitted only on the fitting partition. Test CSVs were not read
in the tuning command. Calibration scores were not used for model selection.

## Settings fixed before the grid ran

- CPU, one PyTorch thread, deterministic PyTorch algorithms.
- Neural seeds: 0, 1, 2, applied before model construction and training.
- Adam, learning rate 0.001, batch size 256, maximum 50 epochs, patience 5.
- MSE reconstruction training; strict tuning-loss improvement; restore best weights.
- Shuffle fitting samples only; chronological partition membership is unchanged.
- MLP hidden widths: 64, 32, mirrored decoder; bottleneck candidates 4, 8, 16, 32.
- PCA full SVD; candidates chosen from fitting explained-variance levels of
  80%, 90%, 95% and ranks four below/above the 90% rank. This yielded
  42, 55, 59, 63, 73 components.
- Operational elbow rule: greatest normalized distance below the line joining
  the smallest/largest candidate's tuning loss. Use seed-mean loss for MLP;
  fall back to minimum loss if there is no positive elbow. This is an explicit
  implementation choice, not a scientifically unique elbow.

## Provisional tuning findings

| Model | Dimension | Mean tuning MSE |
|---|---:|---:|
| PCA | 42 | 0.313895 |
| PCA | 55 | 0.220663 |
| PCA | 59 | 0.193940 |
| PCA | 63 | 0.170455 |
| PCA | 73 | 0.111854 |
| MLP | 4 | 0.757775 |
| MLP | 8 | 0.693692 |
| MLP | 16 | 0.580101 |
| MLP | 32 | 0.441152 |

The elbow rule selected PCA rank 59 and MLP bottleneck 16. The MLP best epochs
were 7, 6 and 11 for seeds 0, 1 and 2. The larger candidates had lower MSE;
the elbow rule deliberately balances dimension against reconstruction improvement.
This is a coarse grid: the selections are sensitive to the candidate range.
PCA and MLP have different capacities here; these numbers do not isolate the
benefit of nonlinearity. Lower tuning reconstruction error does not establish
better attack detection.

## Reproducibility checks

- Repeated selected MLP seed 0 from initialization: identical saved weights and
  tuning scores; matching epoch history within 1e-12 tolerance after CSV reload.
- Repeated selected PCA fit: identical tuning scores.
- All three selected MLP seeds have saved checkpoints and epoch histories.
- Manifest records Python/library versions, platform, seeds, threads, source
  hashes, Git commit/working state, and training CSV hashes.
- This establishes repeatability in this CPU/software environment only. Training
  times are single measurements and naturally vary. No GPU or cross-platform
  reproducibility claim is made.

## How to rerun

Activate the existing project environment, then choose a **new output folder**:

```bash
conda activate dsa4266
python -m ugr16ad.experiments tune --output reports/results/chenlei-week3-repeat
```

The command refuses to overwrite an existing folder. To change the provisional
split, use `--tune-fraction` and a fresh folder, before any test evaluation.

Final evaluation is a separate command using the agreed window length and shared
synthetic check: 1% of calibration rows, seed 0, magnitudes 2 and 4 crossed
with 1 and 5 features, reported as ROC-AUC:

```bash
python -m ugr16ad.experiments evaluate \
  --output reports/results/chenlei-week3-20261026 \
  --window 20
```

Evaluation consumes the saved selected checkpoints, without retraining or
selection on test results. It filters tuning/calibration/test rows to eligible
20-minute-window endpoints within each partition. Calibration thresholds are
99th-percentile scores, then applied unchanged to test. It saves shared-format
scores and feature errors, per-category recall, frozen-threshold metrics,
retrospective ranking metrics, inference timing, and three-seed summaries.
Synthetic shifts apply to a copy of the full standardized calibration data,
before selecting eligible endpoints, with seed 0. The threshold comes only
from untouched calibration scores; synthetic scores never feed back into it. Synthetic
results are a sensitivity sanity check, not evidence of real attack detection.

## Files to keep

`manifest.json`, `grid.json`, `selected.json`, `tuning.csv`, `pca_variance.csv`,
`reproducibility.json`, `tuning-curves.png`, `preprocessor.joblib`, `pca.joblib`,
and `candidates/` with MLP weights and histories. Evaluation adds a separate
`evaluation-window20/` directory. Keep these artifacts outside Git or explicitly
share them with teammates: `reports/results/` is ignored by the repository.

## Remaining work

Run final evaluation with the confirmed protocol above and discuss its results
without retuning against test labels. Existing saved tuning runs can be reused;
no retraining is needed just to update synthetic checks. `sanity_check` is the
shared one from `ugr16ad.synthetic`, the same as every other model uses.


## Running from scratch

Use a new folder to preserve earlier results:

```bash
conda activate dsa4266
python -m pytest
python -m ugr16ad.experiments tune --output reports/results/chenlei-week3-final --tune-fraction 0.5
python -m ugr16ad.experiments evaluate --output reports/results/chenlei-week3-final --window 20
```

`tuning.csv` records the grid and selection inputs. `reproducibility.json`
records repeat-fit checks. `evaluation-window20/comparison.csv` contains per-run
real test metrics; `summary.csv` gives means and sample standard deviations
(neural models only; PCA standard deviation is undefined with one run). Each
model/seed folder also contains `sanity_check.csv` for synthetic calibration
ROC-AUC, separate from real test metrics. Inference times cover scoring already
preprocessed eligible calibration and test rows, excluding preprocessing and model
loading, the same span the other models' run scripts time.


## Shared result layout

Evaluation now saves runs through `results.run_dir` at
`reports/results/pca/seed0/` and `reports/results/mlp/seed{0,1,2}/`. These contain
`scores.csv`, `meta.json`, `feature_errors.npy`, `categories.csv`, and
`sanity_check.csv`, directly accessible to the team's chart/table readers.
The metadata manifest link is relative to each run's actual location.

`--output` remains the checkpoint/experiment folder. Its
`evaluation-window20/` directory holds only protocol and comparison/summary tables
for new runs. `--results-root` optionally selects another model/seed root,
mainly for isolated tests. Existing shared runs are never silently overwritten.
Changing `--output` alone does not change the shared results root.

Previously evaluated results can be copied into the shared model/seed locations
without retraining or rescoring; update the relative manifest path in each copied
`meta.json`. Preserve the original experiment folder as the audit archive.
