"""Model implementations.

Deliberately empty of concrete models for now - the group is still deciding
which families to use. The registry and the :class:`~vulnprio.models.base.Ranker`
contract are in place, so adding one is a new file plus one import line here.

When you add ``src/vulnprio/models/logistic.py``, append::

    from vulnprio.models import logistic  # noqa: F401

below, and the model becomes available to every config and CLI command.
"""

from vulnprio.models.base import Ranker, available_models, get_model, register_model

# Concrete model modules are imported here for their registration side effect.
# (none yet)

__all__ = ["Ranker", "available_models", "get_model", "register_model"]
