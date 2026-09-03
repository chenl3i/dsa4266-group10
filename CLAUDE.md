# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is developed on Windows. `make` is usually unavailable, so invoke the
CLI directly — the `Makefile` recipes document the equivalent commands.

```bash
# Setup (the venv is not committed; create it if missing)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -e .

# Tests — always exclude network-marked tests unless testing acquisition
./.venv/Scripts/python.exe -m pytest -m "not network"
./.venv/Scripts/python.exe -m pytest tests/data/test_splits.py            # one file
./.venv/Scripts/python.exe -m pytest tests/test_leakage.py::TestScrubbing # one class
./.venv/Scripts/python.exe -m pytest -k "embargo"                         # by name
./.venv/Scripts/python.exe -m pytest -m network                           # hits NVD

# Lint (CI runs both; `format --check` fails CI, plain `format` rewrites)
./.venv/Scripts/python.exe -m ruff check src tests
./.venv/Scripts/python.exe -m ruff format src tests

# Pipeline stages
./.venv/Scripts/python.exe -m vulnprio.cli info
./.venv/Scripts/python.exe -m vulnprio.cli fetch --config configs/default.yaml
./.venv/Scripts/python.exe -m vulnprio.cli fetch --year 2015   # one feed file, ~4 MB
```

**Known-failing tests:** 11 tests in `tests/evaluation/test_metrics.py` fail because
`metrics.py` `order_by_score` is an intentionally unimplemented group exercise (see
ADR 0004). Baseline is **73 passed, 11 failed**. Do not treat these as a regression.

**Remaining work** is tracked by stub messages. Every `NotImplementedError` names its
plan task (`docs/superpowers/plans/`), so this is a live progress count:

```bash
grep -rn "NotImplementedError" src/vulnprio
```

## Architecture

Four stages, one CLI command each, **decoupled through the filesystem**. Each stage
writes an artifact and the next stage reads it, so any stage re-runs without repeating
the ones before it — `fetch` takes minutes, `evaluate` takes seconds and runs constantly.

```
configs/*.yaml → [1] fetch → data/raw/<source>/<date>/ + snapshot.json
                → [2] build-dataset → data/interim/analysis_table.parquet
                → [3] build-features → {split: (X, y)}
                → [4] evaluate → reports/results/<experiment>/
```

`docs/architecture.md` names the invariant that must hold at each boundary. Read it
before changing anything that crosses a stage.

### Three things bind the stages together

**`data/schema.py` — the column contract.** `NVD_COLUMNS` is the canonical 20-column
analysis table. Every parser and feature block reads these names, so a parser change
fails `schema.validate` rather than silently breaking feature code downstream.

**`leakage.py` — executable guards, not documentation.** Four leak classes are
enumerated in its docstring; the most important is #3: NVD descriptions are *rewritten*
after a CVE is exploited, so TF-IDF over raw text trains on the answer. `LEAKY_PHRASES`
is stripped before vectorisation. Reference features (#4) are allowed but must be
reported with and without — the ablation is a finding, not an inconvenience.

**Three registries.** Feature blocks (`features/base.py`), models (`models/base.py`)
and baselines (`baselines/registry.py`) are all decorator registries keyed by a `name`
that doubles as the config key. Adding a component is: new file → subclass → decorate →
import in the package `__init__` → enable in config. No pipeline, CLI or evaluation
code changes. `features/cvss.py` is the worked example for blocks.

### Framing that shapes the interfaces

Models are **rankers, not classifiers** (ADR 0001). The deliverable is an ordering
under a patch budget, so `Ranker.score` returns an ordering-quality score and no model
picks a threshold. Reporting accuracy or 0.5-threshold F1 here is the framing error the
interface exists to prevent. `precision@k` across `evaluation.k_percents` is the metric.

Feature blocks subclass scikit-learn's `TransformerMixin` specifically so that anything
fitted (vocabulary, category levels) is fitted on **training rows only** — fitting a
vectoriser before splitting is a leak the Pipeline shape makes hard to commit by accident.

### Temporal correctness

Three dates interact, and the `Config` validator enforces their ordering:

- `split.train_end/valid_end/test_end` — assignment is by **publication date**
- `split.embargo_days` (180) — drops CVEs just before `train_end`, whose exploitation
  status has had less time to be observed; they become `excluded`, not dropped, so the
  count stays visible
- `labels.observation_cutoff` — must be **>= `split.test_end`**. Labels are collected
  past the test boundary because a test CVE exploited six months later is a true
  positive. Restricting collection to the boundary fabricates "recent CVEs are safe".

Anything computed "as of today" is a leak. `features/temporal.py` takes an explicit
reference date for this reason.

### Config

Pydantic with `extra="forbid"`, and `_merge` is **shallow per top-level key**. An
experiment config that overrides one field of a block must restate the whole block —
deliberate, because partially-overridden nested blocks are a silent misconfiguration
source. Experiments live in `configs/experiments/` and use `extends:`.

### Data acquisition (`data/sources/`)

Each source writes `data/raw/<name>/<fetched_on>/` plus a `snapshot.json` manifest.
KEV and EPSS are living datasets, so a result is only reproducible against a named
snapshot. Re-running writes a *new* dated snapshot rather than mutating one.

`NVDSource` has two modes. **`feed`** (default) downloads bulk yearly JSON 2.0 archives —
~172 MB, no API key, and each ships a `.meta` sidecar with a checksum. **`api`** pages
the CVE API 2.0 in 120-day windows; its `build_transport` is still a stub, and it is
only needed for future delta pulls. Both HTTP calls are **injected** (`downloader`,
`transport`) so tests never touch the network.

Two non-obvious facts, both encoded in tests:

- The `.meta` `sha256` is of the **uncompressed JSON**, not the archive — hence separate
  `size` and `zipSize` fields. Hashing the zip fails on every correct download.
- Feeds are partitioned by **CVE ID year, not publication date**, so
  `sources.nvd.feed_year_min/max` (2015–2026) is deliberately **wider** than
  `data.cve_published_min/max` (2016–2024). Narrowing it drops CVEs silently. Stage 2's
  date filter does the real narrowing.

Feeds are stored zipped; read them with stdlib `zipfile` rather than decompressing.

## Conventions

- Docstrings explain **why**, not what — they carry the reasoning that would otherwise
  live only in a PR thread. Match that when adding code.
- Stubs are full signatures with docstrings and a `NotImplementedError` naming the plan
  task. Preserve that shape when adding new ones.
- Tests use tiny synthetic fixtures (`tests/conftest.py`). A test must never depend on a
  downloaded snapshot; network tests are marked and deselected in CI.
- Design decisions go in `docs/decisions/` as ADRs. The four existing ones are load-
  bearing — read them before changing framing, labels, splits or tie-breaking.
- `data/raw`, `data/interim`, `data/processed` and `reports/*` are gitignored;
  `data/external/` is committed.
