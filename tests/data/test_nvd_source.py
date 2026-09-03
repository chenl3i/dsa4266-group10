"""Window arithmetic for the one-time NVD backfill.

The NVD CVE API 2.0 rejects any date-filtered query spanning more than 120
days, so a 2016-2024 pull must be chopped into windows. A bug here does not
raise - it silently drops every CVE in the gap - so the covering property is
tested directly rather than inferred from a record count.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile

import pytest

from vulnprio.data.sources.nvd import MAX_WINDOW_DAYS, date_windows


class TestDateWindows:
    def test_a_short_range_is_a_single_window(self):
        windows = date_windows(dt.date(2024, 1, 1), dt.date(2024, 3, 1))
        assert windows == [(dt.date(2024, 1, 1), dt.date(2024, 3, 1))]

    def test_no_window_exceeds_the_api_limit(self):
        windows = date_windows(dt.date(2016, 1, 1), dt.date(2024, 12, 31))
        assert windows
        for start, end in windows:
            assert (end - start).days <= MAX_WINDOW_DAYS

    def test_windows_are_contiguous_and_cover_the_whole_range(self):
        start, end = dt.date(2016, 1, 1), dt.date(2024, 12, 31)
        windows = date_windows(start, end)
        assert windows[0][0] == start
        assert windows[-1][1] == end
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
            assert next_start == prev_end + dt.timedelta(days=1)

    def test_a_single_day_range_is_allowed(self):
        assert date_windows(dt.date(2024, 5, 5), dt.date(2024, 5, 5)) == [
            (dt.date(2024, 5, 5), dt.date(2024, 5, 5))
        ]

    def test_an_inverted_range_raises(self):
        with pytest.raises(ValueError, match="before"):
            date_windows(dt.date(2024, 12, 31), dt.date(2024, 1, 1))


class FakeTransport:
    """Stands in for the HTTP call, returning canned API 2.0 payloads.

    Records every parameter dict it was called with, so pagination and window
    arithmetic can be asserted without touching the network.
    """

    def __init__(self, total_per_window: int, page_size: int) -> None:
        self.total_per_window = total_per_window
        self.page_size = page_size
        self.calls: list[dict] = []

    def __call__(self, params: dict) -> dict:
        self.calls.append(dict(params))
        start = params["startIndex"]
        n = max(0, min(self.page_size, self.total_per_window - start))
        return {
            "totalResults": self.total_per_window,
            "startIndex": start,
            "resultsPerPage": n,
            "vulnerabilities": [{"cve": {"id": f"CVE-2020-{start + i:04d}"}} for i in range(n)],
        }


@pytest.fixture
def source(tmp_path):
    from vulnprio.data.sources.nvd import NVDSource

    def factory(transport, **kwargs):
        kwargs.setdefault("published_min", dt.date(2024, 1, 1))
        kwargs.setdefault("published_max", dt.date(2024, 3, 1))
        kwargs.setdefault("mode", "api")
        return NVDSource(tmp_path, transport=transport, **kwargs)

    return factory


class TestFetch:
    def test_pages_until_every_result_is_collected(self, source):
        transport = FakeTransport(total_per_window=250, page_size=100)
        snapshot = source(transport).fetch()

        assert [c["startIndex"] for c in transport.calls] == [0, 100, 200]
        assert snapshot.record_count == 250

    def test_writes_one_file_per_page_plus_a_manifest(self, source):
        transport = FakeTransport(total_per_window=250, page_size=100)
        snapshot = source(transport).fetch()

        assert len(snapshot.files) == 3
        for name in snapshot.files:
            assert (snapshot.directory / name).is_file()
        assert (snapshot.directory / "snapshot.json").is_file()

    def test_requests_one_page_sequence_per_date_window(self, source):
        transport = FakeTransport(total_per_window=10, page_size=100)
        source(
            transport, published_min=dt.date(2016, 1, 1), published_max=dt.date(2016, 12, 31)
        ).fetch()

        starts = sorted({c["pubStartDate"][:10] for c in transport.calls})
        # 366 days at a 120-day cap -> four windows, the first at the range start.
        assert len(starts) == 4
        assert starts[0] == "2016-01-01"

    def test_an_empty_window_writes_no_page_file(self, source):
        transport = FakeTransport(total_per_window=0, page_size=100)
        snapshot = source(transport).fetch()

        assert snapshot.files == []
        assert snapshot.record_count == 0


# --- Feed mode ---------------------------------------------------------------

META_2016 = (
    "lastModifiedDate:2026-08-28T03:03:36-04:00\r\n"
    "size:70187186\r\n"
    "zipSize:5159962\r\n"
    "gzSize:5159826\r\n"
    "sha256:3BE4D34AEFBCA187D687B673F5651A540B4EA5A2CE64505E356155340C2600C9\r\n"
)


class TestFeedYears:
    def test_covers_the_configured_range_inclusively(self):
        from vulnprio.data.sources.nvd import feed_years

        assert feed_years(2015, 2026) == list(range(2015, 2027))

    def test_a_single_year_is_allowed(self):
        from vulnprio.data.sources.nvd import feed_years

        assert feed_years(2020, 2020) == [2020]

    def test_an_inverted_range_raises(self):
        from vulnprio.data.sources.nvd import feed_years

        with pytest.raises(ValueError, match="before|<="):
            feed_years(2026, 2015)


class TestParseMeta:
    def test_extracts_the_checksum_despite_crlf(self):
        from vulnprio.data.sources.nvd import parse_meta

        meta = parse_meta(META_2016)
        assert meta["sha256"] == (
            "3BE4D34AEFBCA187D687B673F5651A540B4EA5A2CE64505E356155340C2600C9"
        )

    def test_keeps_colons_inside_the_timestamp_value(self):
        from vulnprio.data.sources.nvd import parse_meta

        # A naive split(":") would truncate this to "2026-08-28T03".
        assert parse_meta(META_2016)["lastModifiedDate"] == "2026-08-28T03:03:36-04:00"

    def test_reads_the_sizes(self):
        from vulnprio.data.sources.nvd import parse_meta

        meta = parse_meta(META_2016)
        assert meta["zipSize"] == "5159962"
        assert meta["size"] == "70187186"


CRLF = chr(13) + chr(10)  # NVD serves .meta with CRLF endings


class FakeDownloader:
    """Serves a real (tiny) zip archive and its .meta, recording every URL.

    Builds an actual zip so the checksum path is exercised faithfully: NVD's
    .meta advertises the sha256 of the *uncompressed* JSON, not of the archive,
    and a fake that hashed the archive bytes would let that bug through.
    ``corrupt_year`` makes one year's payload disagree with its .meta.
    """

    def __init__(self, inner: bytes = b'{"vulnerabilities": []}', corrupt_year: int | None = None):
        self.inner = inner
        self.corrupt_year = corrupt_year
        self.urls: list[str] = []

    def _zip_bytes(self, year: int) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"nvdcve-2.0-{year}.json", self.inner)
        return buffer.getvalue()

    def __call__(self, url: str, destination) -> None:
        self.urls.append(url)
        year = int(url.rsplit("-", 1)[1].split(".")[0])
        if url.endswith(".meta"):
            digest = hashlib.sha256(self.inner).hexdigest().upper()
            if year == self.corrupt_year:
                digest = "0" * 64
            destination.write_text(
                (CRLF).join(
                    [
                        "lastModifiedDate:2026-08-28T03:03:36-04:00",
                        f"size:{len(self.inner)}",
                        f"zipSize:{len(self._zip_bytes(year))}",
                        f"sha256:{digest}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        else:
            destination.write_bytes(self._zip_bytes(year))


@pytest.fixture
def feed_source(tmp_path):
    from vulnprio.data.sources.nvd import NVDSource

    def factory(downloader, **kwargs):
        kwargs.setdefault("published_min", dt.date(2016, 1, 1))
        kwargs.setdefault("published_max", dt.date(2024, 12, 31))
        kwargs.setdefault("mode", "feed")
        kwargs.setdefault("feed_year_min", 2015)
        kwargs.setdefault("feed_year_max", 2016)
        return NVDSource(tmp_path, downloader=downloader, **kwargs)

    return factory


class TestFeedFetch:
    def test_writes_one_zip_and_one_meta_per_year(self, feed_source):
        downloader = FakeDownloader()
        snapshot = feed_source(downloader).fetch()

        names = sorted(snapshot.files)
        assert names == [
            "nvdcve-2.0-2015.json.zip",
            "nvdcve-2.0-2015.meta",
            "nvdcve-2.0-2016.json.zip",
            "nvdcve-2.0-2016.meta",
        ]
        for name in names:
            assert (snapshot.directory / name).is_file()
        assert (snapshot.directory / "snapshot.json").is_file()

    def test_a_checksum_mismatch_raises_and_names_the_year(self, feed_source):
        downloader = FakeDownloader(corrupt_year=2016)

        with pytest.raises(ValueError, match="2016"):
            feed_source(downloader).fetch()

    def test_the_feed_range_is_independent_of_the_study_window(self, feed_source):
        downloader = FakeDownloader()
        feed_source(
            downloader,
            published_min=dt.date(2016, 1, 1),
            published_max=dt.date(2024, 12, 31),
            feed_year_min=2015,
            feed_year_max=2026,
        ).fetch()

        years = sorted({int(u.rsplit("-", 1)[1].split(".")[0]) for u in downloader.urls})
        assert years == list(range(2015, 2027))

    def test_feed_mode_never_touches_the_api_transport(self, feed_source):
        transport = FakeTransport(total_per_window=10, page_size=100)
        feed_source(FakeDownloader(), transport=transport).fetch()

        assert transport.calls == []

    def test_api_mode_still_uses_the_transport(self, feed_source):
        transport = FakeTransport(total_per_window=10, page_size=100)
        feed_source(FakeDownloader(), mode="api", transport=transport).fetch()

        assert transport.calls

    def test_an_unknown_mode_raises(self, feed_source):
        with pytest.raises(ValueError, match="mode"):
            feed_source(FakeDownloader(), mode="carrier-pigeon").fetch()


@pytest.mark.network
def test_the_real_feed_meta_is_reachable_and_parseable(tmp_path):
    """Guards the URL shape against NVD reorganising their feeds.

    Downloads only a ~200-byte .meta, never a feed file.
    """
    from vulnprio.data.sources.nvd import FEED_META_URL, build_downloader, parse_meta

    destination = tmp_path / "2016.meta"
    build_downloader()(FEED_META_URL.format(year=2016), destination)
    meta = parse_meta(destination.read_text(encoding="utf-8"))

    assert len(meta["sha256"]) == 64
    assert int(meta["zipSize"]) > 1_000_000
