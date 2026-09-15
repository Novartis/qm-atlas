"""Collect qm_atlas results into a single flat directory.

Merges the per-conformer result SDF files of every registered state into one
SDF per state, zips the associated ``.cosmo`` files, and copies the per-compound
CSV bookkeeping files (properties, conformer properties, pKa, and screening
results).
"""

import logging
from pathlib import Path
from typing import Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from qm_atlas.command_line.base_config import (
    CalculationInput,
    apply_software_config,
    extract_cpd_dirs,
    setup_logging,
)
from qm_atlas.command_line.file_interface import collect_output

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas collect"


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class CollectInterfaceConfig(BaseModel):
    """Complete configuration for the collect workflow."""

    command: Literal["collect"] = Field(
        default="collect",
        description="CLI command name",
    )
    input: CalculationInput = Field(
        description="Results directory containing compound subdirectories to collect from",
    )
    output_directory: Path | None = Field(
        default=None,
        description=(
            "Directory to write the collected files to. "
            f"Defaults to '{collect_output.OUTPUT_DIR_NAME}' inside the results directory."
        ),
    )
    use_v2000: bool = Field(
        default=False,
        description="Write merged SDF files using the V2000 format instead of V3000",
    )
    software_config: Path | None = Field(
        default=None,
        description="Path to software configuration YAML file (overrides qm_atlas_SOFTWARE_CONFIG_FILE env var)",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> CollectInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        CollectInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Collect Workflow — Gather qm_atlas results into a single directory.",
        env_prefix="qm_atlas_COLLECT_",
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    parser.add_class_arguments(CollectInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = CollectInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def _resolve_target_dir(config: CollectInterfaceConfig) -> Path:
    """Determine the directory the collected files should be written to."""
    if config.output_directory is not None:
        return config.output_directory.expanduser().resolve()

    results_directory = config.input.results_directory
    if results_directory is None:
        raise ValueError(
            "output_directory must be provided when no results_directory is given in input"
        )
    return results_directory / collect_output.OUTPUT_DIR_NAME


def run_local(config: CollectInterfaceConfig) -> None:
    """Run the collect workflow locally.

    Args:
        config: CollectInterfaceConfig with all settings.
    """
    cpd_dirs = extract_cpd_dirs(config.input)
    if not cpd_dirs:
        _logger.warning("No compound directories found to collect from")
        return

    target_dir = _resolve_target_dir(config)
    _logger.info(f"Collecting results from {len(cpd_dirs)} compound directories into {target_dir}")

    collect_output.collect_files(cpd_dirs, target_dir, use_v2000=config.use_v2000)

    _logger.info(f"Finished collecting results into {target_dir}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: CollectInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the collect CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (CollectInterfaceConfig | None):
            Optional CollectInterfaceConfig object. If provided, overrides command-line arguments.
        log_file (Path | None):
            Optional file to append logs to.
    """
    if config is None:
        config = load_config(argv=argv)
    apply_software_config(config.software_config)
    setup_logging(verbose=config.verbose, log_file=log_file)

    run_local(config)


if __name__ == "__main__":
    main()
