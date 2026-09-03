"""Common contract for every external data source.

Each source downloads to ``data/raw/<name>/<fetched_on>/`` and writes a
``snapshot.json`` manifest beside it. Recording *when* data was pulled is not
bookkeeping niceness here: KEV and EPSS are living datasets, so a result is only
reproducible against a named snapshot.
"""

from __future__ import annotations

import abc
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vulnprio.paths import ensure_dir


@dataclass(frozen=True)
class Snapshot:
    """Manifest describing one download of one source."""

    source: str
    fetched_on: dt.date
    directory: Path
    files: list[str] = field(default_factory=list)
    record_count: int | None = None
    notes: str = ""

    def write_manifest(self) -> Path:
        payload = asdict(self)
        payload["fetched_on"] = self.fetched_on.isoformat()
        payload["directory"] = str(self.directory)
        manifest = self.directory / "snapshot.json"
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, directory: Path) -> Snapshot:
        payload = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        payload["fetched_on"] = dt.date.fromisoformat(payload["fetched_on"])
        payload["directory"] = Path(payload["directory"])
        return cls(**payload)


class DataSource(abc.ABC):
    """Downloads one external dataset into the raw layer."""

    #: Short identifier, used as the raw subdirectory name.
    name: str

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root

    def snapshot_dir(self, fetched_on: dt.date | None = None) -> Path:
        fetched_on = fetched_on or dt.date.today()
        return ensure_dir(self.raw_root / self.name / fetched_on.isoformat())

    def latest_snapshot(self) -> Snapshot | None:
        """Most recent snapshot on disk, or ``None`` if the source is unfetched."""
        root = self.raw_root / self.name
        if not root.is_dir():
            return None
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "snapshot.json").exists())
        return Snapshot.load(dirs[-1]) if dirs else None

    @abc.abstractmethod
    def fetch(self, force: bool = False) -> Snapshot:
        """Download the source into a new snapshot directory."""
