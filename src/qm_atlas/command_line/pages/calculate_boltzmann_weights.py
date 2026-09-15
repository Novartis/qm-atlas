"""Calculate per-conformer Boltzmann weights from a calculated energy property.

For every registered state, the energies of its conformers (taken from a named
energy property in the conformer-property table) are converted to Boltzmann
weights at a given temperature. The normalised weights are written back as a new
per-conformer property, both to the conformer-property CSV and to the conformer
SDF files, so they can be consumed by ``aggregate_conformer_properties`` for a
Boltzmann-weighted average.
"""

import logging
import sys
import time
import traceback
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
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.tasks import boltzmann
from qm_atlas.tasks.boltzmann import ENERGY_UNIT_TO_J_PER_MOL
from qm_atlas.tasks.common import ScalarProperty

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas calculate_boltzmann_weights"


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class BoltzmannWeightOptions(BaseModel):
    """Options controlling how Boltzmann weights are calculated."""

    energy_property: str = Field(
        ...,
        description="Name of the per-conformer energy property to weight on "
        "(as written in the conformer SDF files and conformer-property CSV)",
    )
    energy_unit: Literal["hartree", "kcal/mol", "kJ/mol", "eV", "J/mol"] = Field(
        default="hartree",
        description="Unit of the energy property",
    )
    temperature: float = Field(
        default=298.15,
        gt=0,
        description="Temperature in Kelvin for the Boltzmann distribution",
    )
    weight_property: str = Field(
        default="boltzmann_weight",
        description="Name of the per-conformer property to store the calculated weights under",
    )
    relative_energy_unit: Literal["hartree", "kcal/mol", "kJ/mol", "eV", "J/mol"] = Field(
        default="kcal/mol",
        description="Unit in which to report each conformer's energy difference "
        "to the lowest-energy conformer",
    )
    relative_energy_property: str = Field(
        default="relative_energy",
        description="Name of the per-conformer property to store the energy "
        "difference to the lowest-energy conformer under",
    )


class CalculateBoltzmannWeightsInterfaceConfig(BaseModel):
    """Complete configuration for the Boltzmann-weight calculation workflow."""

    command: Literal["calculate_boltzmann_weights"] = Field(
        default="calculate_boltzmann_weights",
        description="CLI command name",
    )
    input: CalculationInput = Field(
        description="Results directory containing compound subdirectories to process",
    )
    boltzmann_options: BoltzmannWeightOptions = Field(
        description="Boltzmann-weight calculation options",
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


def load_config(argv: list[str] | None = None) -> CalculateBoltzmannWeightsInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        CalculateBoltzmannWeightsInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Calculate Boltzmann Weights Workflow — Configure via YAML or CLI arguments.\n"
        f"Supported energy units: {sorted(ENERGY_UNIT_TO_J_PER_MOL.keys())}.",
        env_prefix="qm_atlas_CALCULATE_BOLTZMANN_WEIGHTS_",
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    parser.add_class_arguments(CalculateBoltzmannWeightsInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = CalculateBoltzmannWeightsInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def _read_conformer_energy(cpd_dir: compound_dir.CpdDir, sdf_file: Path, energy_property: str):
    """Return the energy of a conformer, or ``None`` if it is not available."""
    mol = cpd_dir.extract_mol(sdf_file)
    if not mol.HasProp(energy_property):
        return None
    try:
        return mol.GetDoubleProp(energy_property)
    except (ValueError, RuntimeError):
        _logger.warning(
            f"Energy property '{energy_property}' on {sdf_file.name} is not numeric; skipping"
        )
        return None


def run_job(
    cpd_dir: compound_dir.CpdDir,
    options: BoltzmannWeightOptions,
) -> None:
    """Calculate and store Boltzmann weights for every state of a compound directory."""
    for state_name in cpd_dir.registry_handler.get_registered_names():
        result_files = cpd_dir.get_result_files(state_name)
        if not result_files:
            continue

        energies: list[float] = []
        valid_files: list[Path] = []
        for sdf_file in result_files:
            energy = _read_conformer_energy(cpd_dir, sdf_file, options.energy_property)
            if energy is None:
                continue
            energies.append(energy)
            valid_files.append(sdf_file)

        if not energies:
            _logger.warning(
                f"No conformers with energy property '{options.energy_property}' for state "
                f"'{state_name}', skipping"
            )
            continue

        weights = boltzmann.compute_boltzmann_weights(
            energies, options.energy_unit, options.temperature
        )
        relative_energies = boltzmann.compute_relative_energies(
            energies, options.energy_unit, options.relative_energy_unit
        )

        for sdf_file, weight, relative_energy in zip(valid_files, weights, relative_energies):
            new_properties = {
                options.weight_property: ScalarProperty(weight),
                options.relative_energy_property: ScalarProperty(relative_energy),
            }
            cpd_dir.add_properties_to_sdf(sdf_file, new_properties)
            cpd_dir.add_properties_to_conformer_csv(sdf_file, new_properties)

        _logger.info(
            f"Wrote Boltzmann weights and relative energies for {len(valid_files)} "
            f"conformers of state '{state_name}'"
        )


def run_local(config: CalculateBoltzmannWeightsInterfaceConfig) -> None:
    """Run the Boltzmann-weight calculation workflow locally.

    Args:
        config: CalculateBoltzmannWeightsInterfaceConfig with all settings.
    """
    cpd_dirs = extract_cpd_dirs(config.input)
    if not cpd_dirs:
        _logger.warning("No compound directories found to process")
        return

    try:
        for cpd_dir in cpd_dirs:
            cpd_dir.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = cpd_dir.log_dir / "calculate_boltzmann_weights.log"
            task_files.change_log_file(
                log_file,
                filemode="a",
                level=logging.DEBUG if config.verbose else logging.INFO,
                format="%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            start_time = time.perf_counter()
            try:
                _logger.info(f"Calculating Boltzmann weights for {cpd_dir.cpd_name}")
                run_job(cpd_dir, config.boltzmann_options)
            except KeyboardInterrupt:
                _logger.error("Got ^C while calculating Boltzmann weights.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Boltzmann weight calculation failed for {cpd_dir.cpd_name}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())
            finally:
                eval_time = time.perf_counter() - start_time
                _logger.info(f"Calculated Boltzmann weights in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: CalculateBoltzmannWeightsInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the calculate_boltzmann_weights CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (CalculateBoltzmannWeightsInterfaceConfig | None):
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
