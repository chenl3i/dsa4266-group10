"""Risk-based vulnerability prioritisation.

Ranking CVEs by likelihood of real-world exploitation, evaluated as the
decision problem it actually is: given a patch budget of k%, which k% do we
patch first, and do we beat ranking by CVSS severity alone?
"""

__version__ = "0.1.0"
