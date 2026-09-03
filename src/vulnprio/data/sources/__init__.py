"""One module per external data source. All implement :class:`DataSource`."""

from vulnprio.data.sources.base import DataSource, Snapshot

__all__ = ["DataSource", "Snapshot"]
