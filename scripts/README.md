# Scripts

Thin, runnable entry points. Logic that a later step depends on belongs in
`src/ugr16ad/` where it can be tested.

| Script | Purpose |
|---|---|
| `eda_checks.py` | Read-only data checks; source of every number in `docs/eda-findings.md` |
| `run_lstm.py` | Train the LSTM (or `--cell gru`) autoencoder per seed and save `reports/results/<cell>/seed<seed>/` |
| `run_cnn.py` | Train the 1D-CNN autoencoder per seed and save `reports/results/cnn/seed<seed>/` |
| `run_windowed_mlp.py` | Tune-loss sweep (`--sweep`), then train the windowed MLP per seed and save `reports/results/windowed_mlp/seed<seed>/` |
| `make_comparison_table.py` | Read every saved run and write `reports/results/comparison_*.csv` |

PCA and the single-minute MLP run through `python -m ugr16ad.experiments` (see
`docs/chenlei-week3.md`), since that code lives in `src/ugr16ad/experiments.py`.

Run from the repo root after `pip install -e .`:

```bash
python scripts/eda_checks.py
python scripts/run_lstm.py --seed 0 1 2
python scripts/run_cnn.py --seed 0 1 2
python scripts/run_windowed_mlp.py --latent-dim 16 --seed 0 1 2
python scripts/make_comparison_table.py
```
