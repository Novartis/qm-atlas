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
from qm_atlas.command_line.file_interface import compound_dir, input_file

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class ReferenceConformersInputConfig(BaseModel):

    command: Literal["add_reference_conformers"] = Field(
        default="add_reference_conformers",
        description="CLI command name",
    )
    input_file: list[Path] = Field(
        ...,
        description="Input files for the reference conformations (SDF) or directory paths",
    )
    results_directory: Path = Field(
        ...,
        description="Results directory where compound subdirectories will be created (will be created if missing)",
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
            if path.is_file() and path.suffix.lower() != ".sdf":
                raise ValueError(f"Unsupported input file type: {path} (must be .sdf)")

        return paths

    @field_validator("results_directory", mode="after")
    @classmethod
    def validate_results_directory_field(cls, v: Path) -> Path:
        """Validate results directory using arg_utils validation."""
        validate_results_dir(v, create_if_missing=False)
        return v


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> ReferenceConformersInputConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        ReferenceConformersInputConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="ReSCoSS Conformer Generation — Configure via YAML or CLI arguments",
        env_prefix="qm_atlas_RESCOSS_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full ReferenceConformersInputConfig as structured arguments
    parser.add_class_arguments(ReferenceConformersInputConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = ReferenceConformersInputConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Connecting to the File Interface
# ---------------------------------------------------------------------------


def _find_cpd_dirs(results_dir: Path) -> list[compound_dir.CpdDir]:
    """Return all compound directories directly under *results_dir*."""
    cpd_dirs: list[compound_dir.CpdDir] = []
    for sub_dir in results_dir.iterdir():
        if sub_dir.is_dir() and compound_dir.is_cpd_dir(sub_dir):
            cpd_dirs.append(compound_dir.create_cpd_dir(sub_dir, create_new=False))
    return cpd_dirs


def _matches(cpd_dir: compound_dir.CpdDir, mol: Chem.Mol, *, allow_stereo_fallback: bool) -> bool:
    """True if *cpd_dir* has a registered state chemically matching *mol*."""
    try:
        cpd_dir.find_state_by_mol(mol, allow_stereo_fallback=allow_stereo_fallback)
    except ValueError:
        return False
    return True


def _find_owning_cpd_dir(
    cpd_dirs: list[compound_dir.CpdDir], mol: Chem.Mol
) -> compound_dir.CpdDir | None:
    """Return the single compound directory chemically matching *mol*.

    An exact isomeric SMILES match is preferred: only when no directory matches
    exactly is the stereo-insensitive fallback consulted. This prevents a
    stereo-insensitive collision with a sibling stereoisomer (e.g. an epimer
    registered as a separate compound) from masking a perfect match. Returns
    ``None`` when zero or more than one directory matches at the chosen tier, so
    the caller can skip ambiguous molecules loudly.
    """
    exact = [
        cpd_dir for cpd_dir in cpd_dirs if _matches(cpd_dir, mol, allow_stereo_fallback=False)
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    fallback = [
        cpd_dir for cpd_dir in cpd_dirs if _matches(cpd_dir, mol, allow_stereo_fallback=True)
    ]
    if len(fallback) == 1:
        return fallback[0]
    return None


def register_reference_conformers(results_dir: Path, reference_conformers: list[Chem.Mol]):

    cpd_dirs = _find_cpd_dirs(results_dir)

    for mol in reference_conformers:
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
        cpd_dir = _find_owning_cpd_dir(cpd_dirs, mol)
        if cpd_dir is None:
            _logger.error(
                f"Molecule {mol_name} could not be matched to exactly one compound "
                f"directory. Skipping registration."
            )
            continue
        conf_ids = compound_dir.get_conf_ids(mol)
        for conf_id in conf_ids:
            ref_file = cpd_dir.add_reference_input(mol, conf_id=conf_id)
            _logger.info(
                f"Registered reference conformer for molecule {mol_name} in file {ref_file}"
            )


def main(
    argv: list[str] | None = None,
    config: ReferenceConformersInputConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the Reference Conformer Registration CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (ReferenceConformersInputConfig | None):
            Optional ReferenceConformersInputConfig object. If provided, overrides command-line arguments.
        log_file (Path | None):
            Optional file to append logs to.
    """
    if config is None:
        config = load_config(argv=argv)
    apply_software_config(config.software_config)
    setup_logging(verbose=config.verbose, log_file=log_file)

    input_config = input_file.InputConfig(
        override_input_checks=True,
        require_hydrogens=True,
        require_3d=True,
    )

    reference_conformers = input_file.read_all_molecules(config.input_file, input_config)
    register_reference_conformers(
        config.results_directory, reference_conformers=reference_conformers
    )


if __name__ == "__main__":
    main()
