"""The marker for tests whose code is still a starter stub.

Strict xfail on NotImplementedError does three jobs at once: CI stays green
while stubs are empty, a wrong implementation fails on its real assertion, and
a correct one fails as XPASS until its owner deletes the marker, so finished
work cannot hide behind it.
"""

import pytest

todo = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="starter stub not implemented yet; once it passes, delete @todo",
)
