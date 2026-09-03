# DSA4266 Group 10 — Risk-Based Vulnerability Prioritisation

Using machine learning to prioritise software vulnerabilities by their
**likelihood of exploitation**, rather than by severity alone.

> If a company can patch only 10% of its vulnerabilities, which 10% should it
> patch first — and does a learned ranking beat ranking by CVSS score?

How it all fits together: **[docs/architecture.md](docs/architecture.md)** — the four
stages, the artifact crossing each boundary, and the invariant that must hold there.
Start here if you are new to the project.

Full specification: **[docs/project-spec.md](docs/project-spec.md)**

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install -r requirements.txt && pip install -e .
```

```bash
vulnprio info
```

`vulnprio info` prints the resolved config and everything registered — the
fastest check that your environment is wired up correctly.

## Pipeline

Each stage reads the same config and writes to disk, so any stage can be re-run
without repeating the ones before it. That matters: `fetch` takes tens of
minutes, `evaluate` takes seconds, and you will run `evaluate` a hundred times.

```bash
vulnprio fetch --config configs/default.yaml
```

```bash
vulnprio build-dataset --config configs/default.yaml
```

```bash
vulnprio build-features --config configs/default.yaml
```

```bash
vulnprio evaluate --config configs/experiments/baseline_cvss.yaml
```

## Layout

```
configs/          Base config + one YAML per experiment
data/             raw / interim / processed (gitignored) + external (committed)
src/vulnprio/     The package
  data/           Sources, schema, parsing, labels, temporal splits
  features/       Toggleable feature blocks (cvss is the worked example)
  models/         Ranker interface + registry  (no models yet, by design)
  baselines/      CVSS, EPSS, random
  evaluation/     precision@k, recall@k, PR-AUC, bootstrap CIs, figures
  leakage.py      Executable leakage guards
notebooks/        EDA and presentation only — no logic other code depends on
tests/            pytest
docs/decisions/   ADRs: why the project is shaped this way
reports/          Figures and result tables
```

## Key design decisions

Read these before writing code — each one shapes how the rest of the project
fits together.

| ADR | Decision |
|---|---|
| [0001](docs/decisions/0001-project-framing.md) | Ranking under a budget, not classification |
| [0002](docs/decisions/0002-label-definition.md) | KEV as target, with the positive-unlabelled caveat |
| [0003](docs/decisions/0003-temporal-split.md) | Temporal split with a 180-day embargo |
| [0004](docs/decisions/0004-tie-breaking.md) | Tie-breaking is explicit and reported |

## Adding things

**A feature block** — new file in `src/vulnprio/features/`, subclass
`FeatureBlock`, decorate with `@register_block`, import it in
`features/__init__.py`, add it under `features.blocks` in config.
Copy [`features/cvss.py`](src/vulnprio/features/cvss.py) — it is the worked example.

**A model** — see [`src/vulnprio/models/README.md`](src/vulnprio/models/README.md).
Nothing in the pipeline, CLI, or evaluation changes.

## Development

```bash
pytest -m "not network"
```

```bash
ruff check src tests && ruff format src tests
```

## Data sources

| Source | Role | Licence / access |
|---|---|---|
| [NIST NVD](https://nvd.nist.gov/) | Features | Public; API key recommended (`.env`) |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) | Labels | Public |
| [FIRST EPSS](https://www.first.org/epss/) | Baseline | Public |
