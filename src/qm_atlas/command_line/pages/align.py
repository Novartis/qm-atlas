"""Align all result conformations of every state onto a reference conformation.

For every registered state, all result conformers are overlaid onto a single
reference conformation and their (aligned) coordinates are written back to the
result SDF files. The optional per-conformer RMSD to the reference is stored as
a new property on the conformer SDF and in the conformer-property CSV.

The reference conformation is chosen per state, in one of two ways:

- ``reference_input``: use one of the state's registered reference input
  conformations (e.g. a bioactive pose). By default the first registered
  reference input is used; ``reference_name`` selects a specific one by file stem.
- ``dynamic``: pick a conformer from the state's own results using the shared
  :class:`~qm_atlas.command_line.utils.conformer_selection.ConformerSelection`
  (e.g. a ``keep_lowest: 1`` property filter on an energy property selects the
  lowest-energy conformer). The first selected conformer is used as the reference.

The alignment engine itself is provided by the
:mod:`~qm_atlas.tasks.align` task, which dispatches to interchangeable backends
(currently only ``rdkit``).
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated, Any, Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdkit import Chem

from qm_atlas.command_line.base_config import (
    CalculationInput,
    apply_software_config,
    extract_cpd_dirs,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.file_interface.compound_dir import get_conf_ids
from qm_atlas.command_line.utils import format_utils
from qm_atlas.command_line.utils.conformer_selection import ConformerSelection, select_result_files
from qm_atlas.tasks import align
from qm_atlas.tasks.common import ScalarProperty

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas align"


# ---------------------------------------------------------------------------
# Reference-selection configuration
# ---------------------------------------------------------------------------


class ReferenceOptions(BaseModel):
    """Base class for reference-conformation selection strategies.

    Each concrete strategy declares a unique ``mode`` literal that acts as the
    discriminator for (de)serialization within :data:`ReferenceConfig`.
    """

    # Reject unknown keys so we don't silently ignore typos in config files.
    model_config = ConfigDict(extra="forbid")


class DynamicReference(ReferenceOptions):
    """Align onto a conformer selected dynamically from the state's own results."""

    mode: Literal["dynamic"] = Field(default="dynamic")
    selection: ConformerSelection = Field(
        default_factory=ConformerSelection,
        description="How to pick the reference conformer from the state's results, e.g. "
        "keep_lowest: 1 on an energy property for the lowest-energy conformer. The first "
        "selected conformer is used.",
    )


class ReferenceInputConformation(ReferenceOptions):
    """Align onto one of the state's registered reference input conformations."""

    mode: Literal["reference_input"] = Field(default="reference_input")
    reference_name: str | None = Field(
        default=None,
        description="File stem of the registered reference input conformation to align onto; "
        "defaults to the first reference input registered for the state.",
    )


REFERENCE_REGISTRY: dict[str, type[ReferenceOptions]] = {
    "dynamic": DynamicReference,
    "reference_input": ReferenceInputConformation,
}

# Discriminated union of the reference-selection strategies. The ``mode`` literal
# is the discriminator; :func:`deserialize_reference_config` resolves the concrete
# class from a YAML dict, mirroring the backend-selection convention used by the
# tasks (protonation, optimize, cosmo_properties).
ReferenceConfig = Annotated[
    DynamicReference | ReferenceInputConformation,
    Field(discriminator="mode"),
]


def deserialize_reference_config(data: Any) -> ReferenceOptions:
    """Deserialize a reference-selection config from a dict using its ``mode`` field."""
    if isinstance(data, ReferenceOptions):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Cannot deserialize reference config from {type(data)!r}")

    mode = data.get("mode", "dynamic")
    if mode not in REFERENCE_REGISTRY:
        raise ValueError(f"Unknown reference mode: {mode!r}")
    return REFERENCE_REGISTRY[mode](**data)


# ---------------------------------------------------------------------------
# Top-level align options
# ---------------------------------------------------------------------------


class AlignOptions(BaseModel):
    """Options controlling how result conformations are aligned."""

    # Optional[union] with a None default lets jsonargparse resolve a single
    # discriminated-union field from YAML without merging a concrete default's
    # fields; run_job coalesces None to the default strategy / backend.
    reference: ReferenceConfig | None = Field(
        default=None,
        description="Which conformation to align onto (dispatch key: mode). Available modes: "
        "'dynamic' (default), 'reference_input'.",
    )
    alignment: align.AlignmentConfig | None = Field(
        default=None,
        description="Alignment backend and its options (dispatch key: backend). Available "
        "backends: 'rdkit' (default).",
    )
    rmsd_property: str | None = Field(
        default="alignment_rmsd",
        description="Name of the per-conformer property to store each conformer's RMSD "
        "to the reference under. Set to null to skip writing the RMSD.",
    )

    @field_validator("reference", mode="before")
    @classmethod
    def _deserialize_reference(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return deserialize_reference_config(value)
        return value

    @field_validator("alignment", mode="before")
    @classmethod
    def _deserialize_alignment(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return align.deserialize_alignment_config(value)
        return value


class AlignInterfaceConfig(BaseModel):
    """Complete configuration for the conformer-alignment workflow."""

    command: Literal["align"] = Field(
        default="align",
        description="CLI command name",
    )
    input: CalculationInput = Field(
        description="Results directory containing compound subdirectories to process",
    )
    align_options: AlignOptions = Field(
        default_factory=AlignOptions,
        description="Conformer-alignment options",
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


def load_config(argv: list[str] | None = None) -> AlignInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        AlignInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Align Result Conformations — Configure via YAML or CLI arguments.",
        env_prefix="qm_atlas_ALIGN_",
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    parser.add_class_arguments(AlignInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = AlignInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def _resolve_reference_mol(
    cpd_dir: compound_dir.CpdDir,
    state_name: str,
    reference: ReferenceOptions,
) -> Chem.Mol | None:
    """Resolve the reference molecule for a state, or None if unavailable."""
    if isinstance(reference, ReferenceInputConformation):
        reference_inputs = cpd_dir.get_reference_inputs(state_name)
        if not reference_inputs:
            _logger.warning(
                f"State '{state_name}' has no registered reference input conformation; skipping"
            )
            return None
        if reference.reference_name is not None:
            matches = [f for f in reference_inputs if f.stem == reference.reference_name]
            if not matches:
                _logger.warning(
                    f"No reference input named '{reference.reference_name}' for state "
                    f"'{state_name}'; skipping"
                )
                return None
            reference_file = matches[0]
        else:
            reference_file = reference_inputs[0]
    else:
        selected = select_result_files(cpd_dir, state_name, reference.selection)
        if not selected:
            _logger.warning(
                f"Reference selection returned no conformer for state '{state_name}'; skipping"
            )
            return None
        if len(selected) > 1:
            _logger.warning(
                f"Reference selection returned {len(selected)} conformers for state "
                f"'{state_name}'; using '{selected[0].stem}'"
            )
        reference_file = selected[0]

    return cpd_dir.extract_mol(reference_file)


def run_job(
    cpd_dir: compound_dir.CpdDir,
    options: AlignOptions,
) -> None:
    """Align all result conformers of every state of a compound directory."""
    # Coalesce the Optional[union] fields to their defaults (see AlignOptions).
    reference = options.reference if options.reference is not None else DynamicReference()
    alignment = (
        options.alignment if options.alignment is not None else align.RdkitAlignmentOptions()
    )

    for state_name in cpd_dir.registry_handler.get_registered_names():
        result_files = cpd_dir.get_result_files(state_name)
        if not result_files:
            continue

        reference_mol = _resolve_reference_mol(cpd_dir, state_name, reference)
        if reference_mol is None:
            continue

        n_aligned = 0
        for sdf_file in result_files:
            probe_mol = cpd_dir.extract_mol(sdf_file)
            try:
                aligned_mol, rmsd_dict = align.align_to_reference(
                    probe_mol, reference_mol, alignment
                )
            except (RuntimeError, ValueError) as exc:
                _logger.warning(f"Alignment failed for '{sdf_file.name}': {exc}")
                continue

            conf_ids = get_conf_ids(aligned_mol)
            rmsd = rmsd_dict.get(conf_ids[0]) if conf_ids else None
            if options.rmsd_property and rmsd is not None:
                ScalarProperty(float(rmsd)).set_property_on_mol(aligned_mol, options.rmsd_property)

            format_utils.write_sdf(aligned_mol, sdf_file, use_v2000=False, conf_ids=conf_ids)

            if options.rmsd_property and rmsd is not None:
                cpd_dir.add_properties_to_conformer_csv(
                    sdf_file, {options.rmsd_property: ScalarProperty(float(rmsd))}
                )
            n_aligned += 1

        _logger.info(f"Aligned {n_aligned} conformers of state '{state_name}'")


def run_local(config: AlignInterfaceConfig) -> None:
    """Run the conformer-alignment workflow locally.

    Args:
        config: AlignInterfaceConfig with all settings.
    """
    cpd_dirs = extract_cpd_dirs(config.input)
    if not cpd_dirs:
        _logger.warning("No compound directories found to process")
        return

    try:
        for cpd_dir in cpd_dirs:
            cpd_dir.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = cpd_dir.log_dir / "align.log"
            task_files.change_log_file(
                log_file,
                filemode="a",
                level=logging.DEBUG if config.verbose else logging.INFO,
                format="%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            start_time = time.perf_counter()
            try:
                _logger.info(f"Aligning result conformations for {cpd_dir.cpd_name}")
                run_job(cpd_dir, config.align_options)
            except KeyboardInterrupt:
                _logger.error("Got ^C while aligning result conformations.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Aligning result conformations failed for {cpd_dir.cpd_name}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())
            finally:
                eval_time = time.perf_counter() - start_time
                _logger.info(f"Aligned result conformations in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: AlignInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the align CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (AlignInterfaceConfig | None):
            Optional config object. If provided, overrides command-line arguments.
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
