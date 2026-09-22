"""``qm-atlas test_software`` — run the packaged wrapper self-tests on any install.

Runs the shipped ``suite/`` wrapper tests via pytest against a software
configuration. Tools not available per the config are skipped, so a clean run
means "every configured tool works"; a failure means a broken install/license.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_SUITE_DIR = Path(__file__).parent / "suite"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qm-atlas test_software",
        description=(
            "Run the wrapper self-tests against a software configuration. Tools "
            "that are unavailable per the configuration are skipped."
        ),
    )
    parser.add_argument(
        "--software_config",
        default=None,
        help=(
            "Path to the software configuration YAML. If omitted, "
            "QM_ATLAS_SOFTWARE_CONFIG_FILE (or the bundled default) is used."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write a JUnit XML report to this path (for CI).",
    )
    # Remaining args are forwarded to pytest (e.g. -k, -x, -v).
    args, extra = parser.parse_known_args(argv)

    pytest_args = [str(_SUITE_DIR), "-p", "no:cacheprovider"]
    if args.software_config:
        pytest_args += ["--software_config", args.software_config]
    if args.report:
        pytest_args += [f"--junitxml={args.report}"]
    pytest_args += extra

    exit_code = int(pytest.main(pytest_args))
    # Only raise on failure: the dispatcher turns any SystemExit into a non-zero
    # exit, so raising on success (SystemExit(0)) would wrongly report failure.
    if exit_code != 0:
        sys.exit(exit_code)
