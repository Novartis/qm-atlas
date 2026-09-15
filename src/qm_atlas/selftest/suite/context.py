"""Compatibility shim for the packaged wrapper tests.

The tests use ``from context import require_software`` / ``import context``.
When pytest collects the ``suite/`` directory it puts this file on the path, so
those imports resolve here and re-export the real helpers from the package.
"""

from qm_atlas.selftest.support import RESOURCES, get_n_cores, require_software  # noqa: F401
