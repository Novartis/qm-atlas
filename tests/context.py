import functools
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

import pytest
from hpc_funcs.schedulers.uge.environment import get_cores

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import conformer_generation

Function = TypeVar("Function", bound=Callable)

# Set default logging to debug for tests
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Resources
RESOURCES = Path("./tests/resources/")


def pytest_configure():
    pytest.RESOURCES = RESOURCES


def get_n_cores() -> int:
    if os.getenv("SGE_TASK_ID") is None:
        return 1
    else:
        return get_cores()


def create_homedir_tmp_path():
    """ """

    # TODO Auto delete

    # NOTE Important. When submitting molecules, you cannot use node specific
    # directory. You need to use globally avaiable scratch and results.
    user_homedir = Path.home()
    random_name = next(tempfile._get_candidate_names())

    tmp_path = user_homedir / "tmp" / f"pytest_{random_name}"

    # Create tempdir in homedir
    # tmp_ = tempfile.TemporaryDirectory(dir=user_homedir/"tmp", prefix="pytest_")
    # tmp_path = Path(tmp_.name).resolve()
    tmp_path.mkdir(parents=True, exist_ok=True)

    return tmp_path


def require_software(*software_names: str) -> Callable[[Function], Function]:
    """
    Decorator to skip a test if any required software is not available.

    This decorator checks at runtime whether all required software packages are
    available according to the software configuration. If any software is unavailable,
    the test is skipped with a descriptive reason.

    Args:
        *software_names: Names of required software packages (e.g., "turbomole", "xtb")

    Returns:
        A decorator function that wraps the test function

    Example:
        @require_software("turbomole")
        def test_single_point(tmp_path):
            ...

        @require_software("turbomole", "xtb")
        def test_multi_tool_integration(tmp_path):
            ...
    """

    def decorator(func: Function) -> Function:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            software_env_manager = SOFTWARE_CONFIG.get_environment_manager()

            # Check which software is unavailable
            unavailable = [
                name
                for name in software_names
                if not software_env_manager.software_available(name)
            ]

            if unavailable:
                reason = f"Required software not available: {', '.join(unavailable)}"
                pytest.skip(reason)

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


TEST_CONF_GEN_OPTIONS = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        conformer_generation.RdkitOptions(rms_threshold=0.3, max_conformers=1000, method="KDG"),
    ],
    combination_name="joint_set",
)
