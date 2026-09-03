"""FIRST Exploit Prediction Scoring System (EPSS).

EPSS is the published state of the art for exactly this task: a daily-refreshed
probability that a CVE will be exploited in the next 30 days. It is the baseline
that makes the project's result credible - beating CVSS is easy, and a reviewer
will immediately ask how the model compares to the system built for this purpose.

Two important caveats, both of which the evaluation must state plainly:

1. EPSS is trained partly on KEV, so comparing on a KEV-derived target is
   somewhat favourable to it. This is a limitation to disclose, not to hide.
2. Daily score files are published per date. For an honest comparison the score
   must be taken *as of* a date at or before the CVE's split boundary - not
   today's score, which encodes the future.

Implemented in Task 4 of the plan.
"""

from __future__ import annotations

from vulnprio.data.sources.base import DataSource, Snapshot

# Daily CSV.GZ archive: .../epss_scores-YYYY-MM-DD.csv.gz
DAILY_ARCHIVE_URL = "https://epss.empiricalsecurity.com/epss_scores-{date}.csv.gz"


class EPSSSource(DataSource):
    name = "epss"

    def fetch(self, force: bool = False) -> Snapshot:
        raise NotImplementedError("Implemented in plan Task 4 (EPSS acquisition).")
