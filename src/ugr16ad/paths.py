"""Canonical filesystem locations.

Every path in the project resolves through here so that no module ever hard-codes
a relative path and breaks when run from a notebook instead of the repo root.
"""

from __future__ import annotations

from pathlib import Path

# src/ugr16ad/paths.py -> src/ugr16ad -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
# The supplied UGR'16 v2 CSVs. Committed: the fixed copy the proposal is written against.
UGR16_DIR = EXTERNAL_DIR / "ugr16v2"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
