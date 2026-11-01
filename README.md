# DSA4266 Group 10 — Nonlinear and Temporal Autoencoders for Network Attack Detection

Do autoencoders that reconstruct single minutes (MLP) or windows of minutes (LSTM,
1D-CNN) detect attacks in UGR'16 v2 network traffic better than a PCA baseline, at a
threshold frozen before seeing test data?

- Method: **[docs/DSA4266.md](docs/DSA4266.md)** (the final proposal)
- Data checks and the risks they expose: **[docs/eda-findings.md](docs/eda-findings.md)**
- Who owns what, and the schedule: **[docs/team-plan.md](docs/team-plan.md)**

The common pipeline and all models are implemented and merged into `working`. Week 3
is experiments and the report.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt && pip install -e .
python scripts/eda_checks.py
```

## Data

`data/external/ugr16v2/` holds the four supplied CSVs (committed):

| File | Rows | Content |
|---|---|---|
| `UGR16v2.Xtrain.csv` | 98,262 | 134 per-minute flow counters, 2016-03-19 → 05-31 |
| `UGR16v2.Ytrain.csv` | 98,262 | 8 label counts per minute |
| `UGR16v2.Xtest.csv` | 43,200 | same features, 2016-07-28 → 08-26 |
| `UGR16v2.Ytest.csv` | 43,200 | same labels |

`Row` (`YYYYMMDDHHMM`) joins X to Y. It is a key, never a feature. The training file
has no attack minutes, so every model learns background traffic only and flags minutes
it reconstructs badly. Test has 1,364 attack minutes (3.2%).

## Pipeline

Every model runs through the same steps, so a difference in results comes from the
model, not the plumbing:

1. **Split** (`data.py`): chronological. Train rows 0–78,608 fit the models. The rest of
   train is validation, split into a tuning block (early stopping) and a calibration
   block (threshold). Segments break at data gaps.
2. **Preprocess** (`preprocess.py`): log1p, standard scaling and constant-feature
   removal, fit on the fitting rows only.
3. **Window** (`windows.py`): 20-minute windows ending at minute *t* that never cross a
   gap. Single-minute models are scored on the same eligible minutes.
4. **Train and score** (`train.py`, `models/`): MSE autoencoders with early stopping.
   The score is the final minute's reconstruction error.
5. **Evaluate** (`evaluate.py`): threshold at the 99th percentile of calibration scores,
   then frozen. Ranking metrics on test are retrospective only.
6. **Save** (`results.py`): `reports/results/<model>/seed<seed>/` holds `scores.csv` and
   `meta.json`. Tables and charts read only these files.

## Running the models

```bash
python -m ugr16ad.experiments tune --output <dir>                # PCA + MLP
python -m ugr16ad.experiments evaluate --output <dir> --window 20
python scripts/run_lstm.py --seed 0 1 2                           # --cell gru for GRU
python scripts/run_cnn.py --seed 0 1 2
python scripts/run_windowed_mlp.py --latent-dim 16 --seed 0 1 2
python scripts/make_comparison_table.py                           # reads every saved run
```

See [scripts/README.md](scripts/README.md) and [docs/chenlei-week3.md](docs/chenlei-week3.md).

## Results (provisional, 30 Oct; final numbers lock 3 Nov)

Test, 43,181 eligible minutes. Neural models are the mean of 3 seeds.

| Model | Recall | Precision | FPR | AP | Recall@FPR≤1% |
|---|---|---|---|---|---|
| PCA | 0.23 | 0.47 | 0.009 | 0.24 | 0.25 |
| MLP | 0.66 | 0.33 | 0.043 | 0.46 | 0.43 |
| Windowed MLP | 0.57 | 0.39 | 0.029 | 0.43 | 0.34 |
| LSTM | 0.60 | 0.37 | 0.034 | 0.43 | 0.33 |
| GRU | 0.54 | 0.40 | 0.027 | 0.42 | 0.32 |
| 1D-CNN | 0.61 | 0.38 | 0.034 | 0.45 | 0.35 |

Every autoencoder beats PCA. On this data the temporal models do not beat the
single-minute MLP. Test FPR runs above the 1% target because the August test traffic
differs from the May calibration block.

## Layout

```
data/             external (committed) + raw / interim / processed (gitignored)
src/ugr16ad/      The package: data, preprocess, windows, train, evaluate, results, models/
scripts/          Run entry points per model, the comparison table, EDA checks
notebooks/        EDA and presentation only; no logic other code depends on
tests/            pytest on tiny synthetic fixtures; `-m "not slow"` skips real-data checks
docs/             Proposal, EDA findings, team plan
reports/          Results, figures, report and slides (gitignored)
```

## Development

```bash
pytest
ruff check src tests && ruff format src tests
```

CI runs both on pushes to `main` and `working` and on PRs.

The earlier vulnerability-prioritisation project (NVD/KEV/EPSS) is preserved at the git
tag `archive/vulnprio`.
