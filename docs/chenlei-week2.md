# PCA and single-minute MLP

Both models reconstruct one preprocessed minute at a time. They return squared
error per feature and its mean as the anomaly score. A high score is not an
attack decision until the calibration threshold is applied.

## Example usage

Run from the project environment with the package installed (`pip install -e .`).
These are initial settings for development, not selected experimental settings.

```python
from ugr16ad.data import load_dataset
from ugr16ad.preprocess import Preprocessor
from ugr16ad.models.pca import PCAScorer
from ugr16ad.models.mlp import MLPAutoencoder
from ugr16ad.train import set_seed, train_autoencoder, score_autoencoder

parts = load_dataset()
pre = Preprocessor().fit(parts["fit"].X)
X = {name: pre.transform(part.X) for name, part in parts.items()}

# PCA learns a linear subspace from fitting data only.
# Example rank: check that it is below the available feature count.
pca = PCAScorer(n_components=8).fit(X["fit"])
pca_tune_scores, pca_tune_errors = pca.score(X["tune"])
coordinates = pca.transform(X["tune"])
print(pca.explained_variance_ratio_.sum())

# MLP: features -> 64 -> 32 -> 8 -> 32 -> 64 -> features.
# Seed BEFORE constructing the model to control initial weights.
seed = 0
set_seed(seed)
model = MLPAutoencoder(X["fit"].shape[1], hidden_dims=(64, 32), latent_dim=8)
result = train_autoencoder(model, X["fit"], X["tune"], seed=seed)
mlp_tune_scores, mlp_tune_errors = score_autoencoder(result.model, X["tune"])
print(result.history, result.best_epoch, result.seconds)
```

The MLP uses ReLU hidden layers and a linear bottleneck and output. A sigmoid
output would restrict reconstruction to [0, 1], which is inappropriate for
standardised features. Its input and output are `(batch, features)`; windows
are deliberately rejected. PCA uses full SVD without whitening for a
deterministic centred projection.

## Before final experiments

- Select PCA rank using fitting-data explained variance and tuning-block checks,
  following the group's predefined selection rule. Do not pick rank by test scores.
- Treat the MLP widths above as starting settings. Fix a small tuning budget
  before experiments and record the final settings.
- Use tuning data for model selection and early stopping. Use the separate
  calibration block for the threshold, then evaluate the test block with frozen
  models and thresholds.
- For comparisons with window models, restrict scores and labels to the same
  eligible timestamps from the shared windowing code. These models themselves
  score every supplied row; they do not perform timestamp filtering.
- Run the selected MLP with three recorded seeds. Measure PCA fitting time and
  both models' inference time in the experiment runner; training time for the
  MLP is already in `TrainResult.seconds`.
- Save preprocessing, model settings/weights, scores and timing with the shared
  results pipeline. This example does not save artifacts or run final evaluation.

## Checks

```bash
python -m pytest tests/test_pca.py tests/test_mlp.py tests/test_train.py -v
python -m ruff check src tests
python -m ruff format --check src tests
```
