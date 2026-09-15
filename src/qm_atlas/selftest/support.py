"""Helpers for the packaged qm_atlas software self-test suite.

These mirror the small test helpers that live in ``tests/`` for development, but
ship inside the wheel so the wrapper tests can run from an installed package.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Callable, TypeVar

import pytest

from qm_atlas.software_environment import SOFTWARE_CONFIG

Function = TypeVar("Function", bound=Callable)

# Test resources shipped alongside the suite (see suite/resources/).
RESOURCES = Path(__file__).parent / "suite" / "resources"


def get_n_cores() -> int:
    """Number of cores to use: 1 locally, the UGE allocation under a scheduler."""
    if os.getenv("SGE_TASK_ID") is None:
        return 1
    from hpc_funcs.schedulers.uge.environment import get_cores

    return get_cores()


def require_software(*software_names: str) -> Callable[[Function], Function]:
    """Skip a test at runtime if any required software is unavailable.

    Availability is read from the active ``SOFTWARE_CONFIG`` (which the suite's
    ``--software_config`` option can override).
    """

    def decorator(func: Function) -> Function:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = SOFTWARE_CONFIG.get_environment_manager()
            unavailable = [name for name in software_names if not manager.software_available(name)]
            if unavailable:
                pytest.skip(f"Required software not available: {', '.join(unavailable)}")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
