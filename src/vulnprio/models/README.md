# Models

Empty by design — model selection is still open. The contract and registry are
ready, so adding one touches three lines outside the new file itself.

## Adding a model

**1. Create `src/vulnprio/models/<name>.py`:**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from vulnprio.models.base import Ranker, register_model


@register_model
class LogisticRanker(Ranker):
    name = "logistic"
    produces_probabilities = True

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced") -> None:
        self.C = C
        self.class_weight = class_weight
        self._model = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=2000, solver="liblinear"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRanker":
        self._model.fit(X, y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "C": self.C, "class_weight": self.class_weight}
```

**2. Register the import** in `src/vulnprio/models/__init__.py`:

```python
from vulnprio.models import logistic  # noqa: F401
```

**3. Enable it** in your config:

```yaml
models:
  enabled: [logistic]
```

## Notes for whichever models you pick

- **Ranking, not classification.** Implement `score`, never `predict`. There is
  no threshold to tune; the budget sets the cut point.
- **Class imbalance.** At roughly a 0.5% positive rate, prefer `class_weight`
  or `scale_pos_weight` over resampling. SMOTE on TF-IDF features is a
  well-known way to manufacture optimistic results — if you use it, fit it
  inside the CV fold, never before the split.
- **Validation set is for tuning only.** Touch the test split once, at the end.
  Every hyperparameter decision comes from `valid`.
- **Justify the choice in the report.** The brief already commits to explaining
  *why* these models — interpretability and their track record on tabular
  imbalanced security data is the argument to make concrete.
