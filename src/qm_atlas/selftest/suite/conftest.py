"""Pytest configuration for the packaged qm_atlas software self-test suite.

Registers the ``--software_config`` option and re-exports ``RESOURCES`` so the
wrapper tests' ``from conftest import RESOURCES`` keeps working.
"""

from pathlib import Path

from qm_atlas.selftest.support import RESOURCES  # noqa: F401  (re-exported for tests)
from qm_atlas.software_environment import SOFTWARE_CONFIG


def pytest_addoption(parser):
    parser.addoption(
        "--software_config",
        action="store",
        default=None,
        help="Path to a software configuration YAML file for the self-test.",
    )


def pytest_configure(config):
    config_file = config.getoption("--software_config")
    if config_file is not None:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Software config file not found: {config_path}")
        SOFTWARE_CONFIG.reload_environment_manager(config_path)
