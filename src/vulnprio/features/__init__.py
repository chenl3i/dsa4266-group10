"""Feature engineering, organised as independently toggleable blocks."""

# Import for side effect: each module registers its block on import.
from vulnprio.features import cpe, cvss, cwe, references, temporal, text  # noqa: F401,E402
from vulnprio.features.base import FeatureBlock, available_blocks, get_block, register_block

__all__ = ["FeatureBlock", "available_blocks", "get_block", "register_block"]
