"""NIST National Vulnerability Database.

Source of the feature side of the dataset: CVSS v3.1 vectors, CWE weakness
mappings, CPE applicability statements, reference URLs, publication dates and
English descriptions.

Scope: a **one-time backfill**. There is no delta or refresh path - re-running
writes a new dated snapshot rather than topping one up, so an earlier result
stays reproducible against the snapshot it was built from.

Two acquisition modes, selected by ``sources.nvd.mode``:

``feed`` (default)
    Bulk JSON 2.0 year-files from ``nvd.nist.gov/feeds/json/cve/2.0/``. About
    172MB for 2015-2026, a few minutes, no API key, and each archive ships a
    ``.meta`` sidecar carrying a sha256 - so a corrupt download is *detected*
    rather than silently parsed. This is the right mode for the backfill.

``api``
    The CVE API 2.0 (``services.nvd.nist.gov/rest/json/cves/2.0``), paged at
    2000 records per request and rate-limited (5 requests / 30s anonymous,
    50 / 30s with an API key from ``NVD_API_KEY``). Date-filtered queries are
    capped at a 120-day span, so the range is chopped by :func:`date_windows`.
    Kept for later incremental pulls; :func:`build_transport` is still a stub.

Both HTTP calls are injected (``downloader``, ``transport``) rather than made
inline. That keeps the paging and checksum logic testable without a network,
and isolates the one piece that has to cope with NVD being unreliable.

Note on checksums: the ``sha256`` in a ``.meta`` file is of the **uncompressed**
JSON, not of the archive - which is why ``size`` and ``zipSize`` are published
separately. See :meth:`NVDSource._verify_checksum`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vulnprio.data.sources.base import DataSource, Snapshot

log = logging.getLogger(__name__)

API_ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 2000

FEED_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.zip"
FEED_META_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.meta"

#: The API rejects a pubStartDate/pubEndDate span wider than this.
MAX_WINDOW_DAYS = 120

#: NVD asks for ~6s between requests even inside the documented rate limit.
DEFAULT_DELAY_SECONDS = 6.0

#: A callable taking API query parameters and returning the decoded JSON body.
Transport = Callable[[dict[str, Any]], dict[str, Any]]

#: A callable downloading ``url`` to ``destination``.
Downloader = Callable[[str, Path], None]


def date_windows(
    start: dt.date, end: dt.date, max_days: int = MAX_WINDOW_DAYS
) -> list[tuple[dt.date, dt.date]]:
    """Chop an inclusive date range into contiguous windows the API will accept.

    Windows are inclusive at both ends and adjacent windows differ by one day,
    so every date in ``[start, end]`` falls in exactly one window. A gap here
    would drop CVEs silently rather than raising, which is why it is tested.
    """
    if end < start:
        raise ValueError(f"start ({start}) must be on or before end ({end})")

    windows: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + dt.timedelta(days=max_days), end)
        windows.append((cursor, window_end))
        cursor = window_end + dt.timedelta(days=1)
    return windows


def feed_years(year_min: int, year_max: int) -> list[int]:
    """The yearly feed files to download, inclusive of both ends.

    Deliberately **not** derived from ``data.cve_published_min/max``. Feeds are
    partitioned by CVE *ID* year, not publication date: a CVE-2015-xxxx first
    published in 2016 lives in the 2015 file, and a CVE-2025-xxxx published in
    2024 lives in the 2025 file. Narrowing the download to the study window
    would drop those silently - no error, just missing rows. So the range is
    configured wider and Stage 2's publication-date filter does the narrowing.
    """
    if year_min > year_max:
        raise ValueError(f"year_min ({year_min}) must be <= year_max ({year_max})")
    return list(range(year_min, year_max + 1))


def parse_meta(text: str) -> dict[str, str]:
    """Parse a ``.meta`` sidecar into its key/value pairs.

    Split on the *first* colon only: ``lastModifiedDate`` carries a timestamp
    with its own colons (``2026-08-28T03:03:36-04:00``), and the files are
    served with CRLF line endings.
    """
    meta: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


class NVDSource(DataSource):
    """One-time backfill of the NVD CVE API into a dated raw snapshot."""

    name = "nvd"

    def __init__(
        self,
        raw_root,
        published_min: dt.date,
        published_max: dt.date,
        *,
        mode: str = "feed",
        feed_year_min: int = 2015,
        feed_year_max: int = 2026,
        api_key: str | None = None,
        transport: Transport | None = None,
        downloader: Downloader | None = None,
        results_per_page: int = RESULTS_PER_PAGE,
    ) -> None:
        super().__init__(raw_root)
        self.published_min = published_min
        self.published_max = published_max
        self.mode = mode
        self.feed_year_min = feed_year_min
        self.feed_year_max = feed_year_max
        self.api_key = api_key if api_key is not None else os.environ.get("NVD_API_KEY")
        self.results_per_page = results_per_page
        self._transport = transport
        self._downloader = downloader

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = build_transport(self.api_key)
        return self._transport

    @property
    def downloader(self) -> Downloader:
        if self._downloader is None:
            self._downloader = build_downloader()
        return self._downloader

    def fetch(self, force: bool = False) -> Snapshot:
        """Run the configured acquisition mode.

        ``feed`` is the one-time backfill; ``api`` remains for later deltas.
        """
        if self.mode == "feed":
            return self._fetch_feed()
        if self.mode == "api":
            return self._fetch_api()
        raise ValueError(f"unknown NVD acquisition mode {self.mode!r}; expected 'feed' or 'api'")

    def _fetch_feed(self) -> Snapshot:
        """Download the yearly JSON 2.0 feed files, verifying each checksum.

        Files are kept zipped: 172MB against 2.08GB decompressed, and Stage 2
        reads the JSON straight out of the archive with stdlib ``zipfile``.
        """
        directory = self.snapshot_dir()
        years = feed_years(self.feed_year_min, self.feed_year_max)
        log.info("NVD feed backfill: %d year file(s), %d-%d", len(years), years[0], years[-1])

        files: list[str] = []
        for year in years:
            meta_name = f"nvdcve-2.0-{year}.meta"
            zip_name = f"nvdcve-2.0-{year}.json.zip"

            meta_path = directory / meta_name
            self.downloader(FEED_META_URL.format(year=year), meta_path)
            meta = parse_meta(meta_path.read_text(encoding="utf-8"))

            zip_path = directory / zip_name
            self.downloader(FEED_URL.format(year=year), zip_path)
            self._verify_checksum(year, zip_path, meta)

            files.extend([meta_name, zip_name])
            log.info("  %s verified (%.1f MB)", zip_name, zip_path.stat().st_size / 1e6)

        snapshot = Snapshot(
            source=self.name,
            fetched_on=dt.date.today(),
            directory=directory,
            files=files,
            # Left unset: the sha256 above already proves each file is intact,
            # and counting would mean decompressing 2GB for a number that
            # Stage 2 reports anyway.
            record_count=None,
            notes=(
                f"One-time bulk feed backfill, CVE ID years {years[0]}-{years[-1]}. "
                "Feeds are partitioned by CVE ID year, so this range is wider than "
                "the study's publication window; Stage 2 applies the date filter."
            ),
        )
        snapshot.write_manifest()
        return snapshot

    @staticmethod
    def _verify_checksum(year: int, path: Path, meta: dict[str, str]) -> None:
        """Check the archive against its ``.meta``.

        The advertised ``sha256`` is of the **uncompressed JSON**, not of the
        zip - which is why ``size`` and ``zipSize`` are published separately.
        Hashing the archive bytes instead fails on every correct download.

        The member is hashed in chunks: the largest year decompresses to ~300MB
        and there is no need to hold it whole just to checksum it.
        """
        expected = meta.get("sha256", "").lower()
        if not expected:
            raise ValueError(f"{year} feed: .meta carries no sha256, cannot verify download")

        zip_size = meta.get("zipSize")
        if zip_size and path.stat().st_size != int(zip_size):
            raise ValueError(
                f"{year} feed: archive is {path.stat().st_size} bytes, .meta says {zip_size}. "
                "The download is truncated; delete the snapshot and re-run."
            )

        digest = hashlib.sha256()
        with zipfile.ZipFile(path) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)

        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"{year} feed: checksum mismatch on {member} - expected {expected}, "
                f"got {actual}. The download is corrupt; delete the snapshot and re-run."
            )

    def _fetch_api(self, force: bool = False) -> Snapshot:
        """Page through every window and write the raw JSON pages to disk.

        Pages are written verbatim, not parsed: parsing belongs to Stage 2, and
        keeping the raw bytes means a parser bug is fixable without a re-pull.
        """
        directory = self.snapshot_dir()
        windows = date_windows(self.published_min, self.published_max)
        log.info(
            "NVD backfill %s to %s: %d window(s)",
            self.published_min,
            self.published_max,
            len(windows),
        )

        files: list[str] = []
        total = 0

        for window_index, (start, end) in enumerate(windows):
            start_index = 0
            while True:
                params = self._params(start, end, start_index)
                payload = self.transport(params)

                vulnerabilities = payload.get("vulnerabilities", [])
                if vulnerabilities:
                    name = f"cves_{start.isoformat()}_{end.isoformat()}_{start_index:07d}.json"
                    (directory / name).write_text(
                        json.dumps(payload, indent=None), encoding="utf-8"
                    )
                    files.append(name)
                    total += len(vulnerabilities)

                # Advance by what the server actually returned, not by what was
                # asked for: NVD may return a short page, and trusting the
                # request size would then skip records silently.
                returned = len(vulnerabilities)
                if returned == 0:
                    break

                total_results = int(payload.get("totalResults", 0))
                start_index += returned
                if start_index >= total_results:
                    break

            log.info(
                "  window %d/%d (%s to %s): %d records so far",
                window_index + 1,
                len(windows),
                start,
                end,
                total,
            )

        snapshot = Snapshot(
            source=self.name,
            fetched_on=dt.date.today(),
            directory=directory,
            files=files,
            record_count=total,
            notes=(
                f"One-time API 2.0 backfill of CVEs published {self.published_min} "
                f"to {self.published_max}, in {len(windows)} window(s)."
            ),
        )
        snapshot.write_manifest()
        return snapshot

    def _params(self, start: dt.date, end: dt.date, start_index: int) -> dict[str, Any]:
        """Query parameters for one page of one window."""
        return {
            "pubStartDate": f"{start.isoformat()}T00:00:00.000",
            "pubEndDate": f"{end.isoformat()}T23:59:59.999",
            "resultsPerPage": self.results_per_page,
            "startIndex": start_index,
        }


def build_downloader(chunk_bytes: int = 1 << 20) -> Downloader:
    """Return the real HTTP downloader for the bulk feed files.

    Streamed in chunks rather than held in memory: the largest year decompresses
    to ~300MB and there is no reason to hold even the 25MB archive whole.

    No retry policy here, unlike the API transport. These are a dozen plain
    static file downloads with no rate limit, and every one is checksum-verified
    against its ``.meta`` - a bad download raises rather than passing silently,
    so the honest fix is to re-run rather than to retry blindly mid-stream.
    """
    import requests

    def _download(url: str, destination: Path) -> None:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_bytes):
                    fh.write(chunk)

    return _download


def build_transport(api_key: str | None) -> Transport:
    """Return the real HTTP transport for the NVD API.

    TODO(group): implement the retry/backoff policy.

    This is the only part of the fetch that has to survive NVD being flaky, and
    the trade-offs are real ones for a 40-minute pull:

      - How many attempts before giving up? Too few and a routine 503 kills the
        run; too many and a genuine outage wastes an hour before you find out.
      - Which failures are retryable? 503 and 403 (NVD returns 403 for rate-limit
        overruns, not 429) are. A 404 or a malformed body is not - retrying those
        just hides a bug in the query.
      - How long to wait between attempts? ``tenacity`` is already a dependency;
        ``wait_exponential`` is the usual choice, but note that NVD's limit is a
        rolling 30-second window, so a backoff shorter than that cannot clear it.
      - Should a partial pull still write a snapshot? Writing one means a later
        parse silently sees a truncated corpus; raising means losing 30 minutes
        of successful pages. Whichever you pick, ``Snapshot.notes`` should say so.

    Sketch to fill in::

        import requests
        from tenacity import retry, stop_after_attempt, wait_exponential

        session = requests.Session()
        if api_key:
            session.headers["apiKey"] = api_key

        @retry(...)                      # <- your policy here
        def _get(params: dict) -> dict:
            response = session.get(API_ENDPOINT, params=params, timeout=60)
            response.raise_for_status()
            return response.json()

        return _get
    """
    raise NotImplementedError(
        "NVD HTTP transport not yet implemented - see the retry-policy notes in "
        "build_transport(). Tests inject a fake transport and do not need this."
    )
