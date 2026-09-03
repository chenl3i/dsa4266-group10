# Notebooks

Notebooks are for **exploration and presentation**, never for logic other code
depends on. Anything a later step needs belongs in `src/vulnprio/` where it can
be imported, tested, and reviewed in a diff.

## Conventions

- Numbered by stage: `01_eda.ipynb`, `02_baselines.ipynb`, ...
- Every notebook starts with the same header cell:

```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, "../src")

from vulnprio.config import load_config
from vulnprio.logging_utils import setup_logging

setup_logging()
cfg = load_config("../configs/default.yaml")
```

- Figures are **exported** to `reports/figures/`, not left inline. A figure that
  exists only inside a notebook cannot go in the report.
- Restart-and-run-all before committing. A notebook that only runs in the order
  you happened to click is not a result anyone can reproduce.
- `data/raw/` is gitignored, so a notebook must never assume a teammate has the
  same files. Read through `vulnprio.paths`, and fail with a clear message
  pointing at `vulnprio fetch`.

## Planned

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Q1-Q4 of the spec: exploitation rate by CVSS band, attack vector, complexity, CWE class, vendor, year |
| `02_baselines.ipynb` | CVSS / EPSS / random comparison, tie-policy bracket |
| `03_features.ipynb` | Block-by-block ablation |
| `04_results.ipynb` | Final figures for the report |
