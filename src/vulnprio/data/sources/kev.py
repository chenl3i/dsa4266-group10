"""CISA Known Exploited Vulnerabilities catalogue.

Source of the positive class. A single ~1MB JSON/CSV file, refreshed as CISA
adds entries.

The field that matters most is ``dateAdded``: the date CISA published the entry,
which is the only timestamp available for when exploitation became *known*.
It is not the date exploitation began, and the gap between the two is why the
temporal split needs an embargo. See docs/decisions/0002-label-definition.md.

Implemented in Task 3 of the plan.
"""

from __future__ import annotations

from vulnprio.data.sources.base import DataSource, Snapshot

CATALOG_JSON_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
)


class KEVSource(DataSource):
    name = "kev"

    def fetch(self, force: bool = False) -> Snapshot:
        raise NotImplementedError("Implemented in plan Task 3 (KEV acquisition).")
