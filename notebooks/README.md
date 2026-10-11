# Notebooks

Notebooks are for **exploration and presentation**, never for logic other code
depends on. Anything a later step needs belongs in `src/ugr16ad/` where it can
be imported, tested, and reviewed in a diff.

## Conventions

- Numbered by stage: `01_eda.ipynb`, `02_pca.ipynb`, ...
- Every notebook starts with the same header cell:

```python
%load_ext autoreload
%autoreload 2

from ugr16ad.logging_utils import setup_logging
from ugr16ad.paths import UGR16_DIR

setup_logging()
```

- Figures are **exported** to `reports/figures/`, not left inline. A figure that
  exists only inside a notebook cannot go in the report.
- Restart-and-run-all before committing. A notebook that only runs in the order
  you happened to click is not a result anyone can reproduce.
- Read data through `ugr16ad.paths`, never a hard-coded relative path.
- Never fit anything (scaler, PCA, threshold) on validation or test rows, even
  for a quick look. Test labels are for characterising the dataset only.
