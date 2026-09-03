"""Reference rankers every model is measured against.

Baselines are :class:`~vulnprio.models.base.Ranker` implementations like any
other model, so the evaluation code cannot accidentally treat them differently.
They live in their own package only because they are *not* learned from the
training data and should not appear in ``models.enabled``.

Three of them, deliberately:

``random``
    The floor. Precision@k for random ranking equals the base rate, so it makes
    the imbalance visible in the same table as everything else.

``cvss``
    The strategy in the project brief, and the thing the research question is
    literally about beating.

``epss``
    The published state of the art for this exact task. Beating CVSS is easy;
    the comparison against EPSS is what makes the result worth reporting.
"""

from vulnprio.baselines.cvss import CVSSBaseline
from vulnprio.baselines.epss import EPSSBaseline
from vulnprio.baselines.random_ranker import RandomBaseline
from vulnprio.baselines.registry import available_baselines, get_baseline, register_baseline

__all__ = [
    "CVSSBaseline",
    "EPSSBaseline",
    "RandomBaseline",
    "available_baselines",
    "get_baseline",
    "register_baseline",
]
