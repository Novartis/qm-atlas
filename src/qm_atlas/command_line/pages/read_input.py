"""Command-line interface for reading input molecules and creating compound directories."""

import logging
from pathlib import Path
from typing import Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem

from qm_atlas.command_line.base_config import (
    apply_software_config,
    setup_logging,
    validate_results_dir,
)
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import (
    PARENT_KEY_PROP,
    InputConfig,
    read_all_molecules,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class ReadInputConfig(BaseModel):
    """Complete configuration for reading input molecules and creating compound directories."""

    command: Literal["read_input"] = Field(
        default="read_input",
        description="CLI command name",
    )
    input_file: list[Path] = Field(
        ...,
        description="Input files (SDF, CSV) or directory paths",
    )
    results_directory: Path = Field(
        ...,
        description="Results directory where compound subdirectories will be created (will be created if missing)",
    )
    input_config: InputConfig = Field(
        default_factory=InputConfig,
        description="Input file reading options (CSV format, validation rules)",
    )
    software_config: Path | None = Field(
        default=None,
        description="Path to software configuration YAML file (overrides qm_atlas_SOFTWARE_CONFIG_FILE env var)",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )

    @field_validator("input_file", mode="after")
    @classmethod
    def validate_input_file(cls, v: list[Path]) -> list[Path]:
        """Resolve paths and validate existence and file type."""
        paths = [p.expanduser().resolve() for p in v]

        for path in paths:
            if not path.exists():
                raise ValueError(f"Input path does not exist: {path}")
            if path.is_file() and path.suffix.lower() not in {".sdf", ".csv"}:
                raise ValueError(f"Unsupported input file type: {path} (must be .sdf or .csv)")

        return paths

    @field_validator("results_directory", mode="after")
    @classmethod
    def validate_results_directory_field(cls, v: Path) -> Path:
        """Validate results directory using arg_utils validation.

        For read_input, the results directory is created if it doesn't exist.
        Pydantic handles conversion to Path.
        """
        validate_results_dir(v, create_if_missing=True)
        return v


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> ReadInputConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        ReadInputConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        env_prefix="qm_atlas_READ_INPUT_",
        description="Read input molecules and create compound directories",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    parser.add_class_arguments(ReadInputConfig)

    args = parser.parse_args(argv)
    config_namespace = parser.instantiate_classes(args)
    return ReadInputConfig(**config_namespace)


def create_compound_directories(
    mols: list[Chem.Mol],
    results_dir: Path,
) -> dict[str, Path]:
    """Create compound directories for each molecule.

    When a molecule carries the :data:`PARENT_KEY_PROP` property (set by the
    input readers when ``parent_col`` / ``parent_property`` is configured),
    molecules sharing the same value are grouped into a single compound
    directory named after that parent key, and each molecule is registered
    as a separate state inside it. Otherwise, each molecule becomes its own
    compound directory named after its ``_Name`` property.

    Args:
        mols: List of molecules
        results_dir: Root directory for compound directories (must exist)

    Returns:
        Dictionary mapping compound directory names to their paths.
    """
    results_dir = Path(results_dir)

    # Group molecules by parent key; molecules without the property are
    # treated as standalone compounds keyed by their own _Name.
    groups: dict[str, list[Chem.Mol]] = {}
    for mol in mols:
        mol_name = mol.GetProp("_Name")
        if mol.HasProp(PARENT_KEY_PROP):
            parent_key = mol.GetProp(PARENT_KEY_PROP)
        else:
            parent_key = mol_name
        groups.setdefault(parent_key, []).append(mol)

    compound_dirs: dict[str, Path] = {}

    for parent_key, group_mols in groups.items():
        cpd_dir_path = results_dir / parent_key
        cpd_dir = create_cpd_dir(cpd_dir_path, create_new=True)

        # Detect duplicate state names within a group early to fail loudly
        # rather than partially registering states.
        seen_names: set[str] = set()
        for mol in group_mols:
            mol_name = mol.GetProp("_Name")
            if mol_name in seen_names:
                _logger.error(
                    f"Duplicate state name '{mol_name}' within parent group "
                    f"'{parent_key}'; skipping duplicate."
                )
                continue
            seen_names.add(mol_name)

            try:
                cpd_dir.add_input_structure(mol, overwrite=True)
            except (OSError, ValueError) as e:
                _logger.error(
                    f"Failed to register state '{mol_name}' in compound directory "
                    f"'{parent_key}': {e}"
                )
                continue

        compound_dirs[parent_key] = cpd_dir_path

    return compound_dirs


# ---------------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------------


def run(config: ReadInputConfig) -> dict[str, Path]:
    """Execute the workflow: read molecules, validate, and create compound directories.

    Args:
        config: ReadInputConfig with all settings

    Returns:
        Dictionary mapping molecule names to their compound directory paths
    """
    _logger.info(
        f"Starting molecule input processing. Results directory: {config.results_directory.resolve()}"
    )
    _logger.debug(f"Configuration:\n{config.model_dump_json(indent=2)}")

    # Read molecules from input files/directories
    all_mols = read_all_molecules(config.input_file, config.input_config)
    _logger.info(f"Read {len(all_mols)} molecules from {len(config.input_file)} input source(s)")

    if len(all_mols) == 0:
        _logger.warning("No valid molecules found after filtering")
        return {}

    # Create compound directories
    _logger.info(f"Creating compound directories for {len(all_mols)} molecules")
    compound_dirs = create_compound_directories(all_mols, config.results_directory)
    return compound_dirs


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: ReadInputConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the read_input CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (ReadInputConfig | None):
            Optional ReadInputConfig object. If provided, overrides command-line arguments.
        log_file (Path | None):
            Optional file to append logs to.
    """
    if config is None:
        config = load_config(argv=argv)

    apply_software_config(config.software_config)
    setup_logging(verbose=config.verbose, log_file=log_file)

    run(config)


if __name__ == "__main__":
    main()
