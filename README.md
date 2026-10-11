# DSA4266 Group 10 — Nonlinear and Temporal Autoencoders for Network Attack Detection

Do autoencoders that reconstruct single minutes (MLP) or windows of minutes (LSTM,
1D-CNN) detect attacks in UGR'16 v2 network traffic better than a PCA baseline, at a
threshold frozen before seeing test data?

- Method: **[docs/DSA4266.md](docs/DSA4266.md)** (the final proposal)
- Data checks and the risks they expose: **[docs/eda-findings.md](docs/eda-findings.md)**

No pipeline or model code exists yet. The repo holds the data, the EDA script, path and
logging helpers, and the tests that keep them honest.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install -r requirements.txt && pip install -e .
```

```bash
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

`Row` (`YYYYMMDDHHMM`) joins X to Y. It is a key, never a feature.

## Layout

```
data/             external (committed) + raw / interim / processed (gitignored)
src/ugr16ad/      The package: paths and logging helpers so far
scripts/          eda_checks.py: read-only data checks behind docs/eda-findings.md
notebooks/        EDA and presentation only; no logic other code depends on
tests/            pytest
docs/             Proposal and EDA findings
reports/          Figures and result tables (gitignored)
```

## Development

```bash
pytest
```

```bash
ruff check src tests && ruff format src tests
```

The earlier vulnerability-prioritisation project (NVD/KEV/EPSS) is preserved at the git
tag `archive/vulnprio`.
