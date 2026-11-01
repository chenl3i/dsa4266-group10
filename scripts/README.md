# Scripts

Thin, runnable entry points. Logic that a later step depends on belongs in
`src/ugr16ad/` where it can be tested.

| Script | Purpose |
|---|---|
| `eda_checks.py` | Read-only data checks; source of every number in `docs/eda-findings.md` |
| `run_lstm.py` | Train the LSTM (or `--cell gru`) autoencoder per seed and save `reports/results/<cell>/seed<seed>/` |
| `sweep.py` | Blind validation sweep (`--model lstm/gru/cnn/windowed_mlp/mlp --stage 1/2/3`, or `--summary-only`) of the run scripts' configs into `reports/runs-sweep/<model>/`; writes `reports/runs-sweep/<model>/summary.csv` (the ablation table) |
| `run_cnn.py` | Train the 1D-CNN autoencoder per seed and save `reports/results/cnn/seed<seed>/` |
| `run_mlp.py` | Train the single-minute MLP autoencoder per seed (one config per call, for the sweep) and save `reports/runs-final/mlp/seed<seed>/` by default |
| `run_windowed_mlp.py` | Tune-loss sweep (`--sweep`), then train the windowed MLP per seed and save `reports/results/windowed_mlp/seed<seed>/` |
| `make_comparison_table.py` | Read every saved run and write `reports/results/comparison_*.csv` |

PCA and Chen Lei's recorded MLP checkpoint/elbow protocol run through
`python -m ugr16ad.experiments` (see `docs/chenlei-week3.md`; code in
`src/ugr16ad/experiments.py`). `run_mlp.py` runs one MLP config through the shared
pipeline, for the sweep and the final runs.

Run from the repo root after `pip install -e .`:

```bash
python scripts/eda_checks.py
python scripts/run_lstm.py --seed 0 1 2
python scripts/run_cnn.py --seed 0 1 2
python scripts/run_windowed_mlp.py --latent-dim 16 --seed 0 1 2
python scripts/make_comparison_table.py
```
