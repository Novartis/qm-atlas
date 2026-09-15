"""Aggregate per-conformer properties into per-state values.

For every registered state, the per-conformer properties calculated by
``conformer_properties`` / ``cosmotherm_properties`` are aggregated over a
selected subset of conformers into a single per-state value and written to the
compound's ``{compound}_properties.csv`` file.

The user names the properties to aggregate directly (rather than re-declaring
the calculation task blocks). Use ``qm-atlas conformer_properties
--print-output-names --config <file>`` (or the ``cosmotherm_properties``
equivalent) to look up the exact SDF property tags a calculation produces.

Supported property types (``property_type``):

- ``scalar``: a single number per conformer.
- ``tensor``: an array per conformer (aggregated element-wise).
- ``atom_based``: one value per atom (aggregated element-wise, per atom).

Supported aggregation operations (all applied over the selected conformers;
element-wise for tensor / atom-based properties):

- ``mean``: unweighted arithmetic mean.
- ``boltzmann_mean``: Boltzmann-weighted mean, using a per-conformer weight
  property (e.g. one produced by ``calculate_boltzmann_weights``).
- ``min`` / ``max``: element-wise minimum / maximum.

Which conformers are aggregated over is controlled by the shared
:class:`~qm_atlas.command_line.utils.conformer_selection.ConformerSelection`
(e.g. reference results only, or an energy window above the lowest-energy
conformer).

Optionally (``representative_structure.enabled: true``), one representative
structure per state is written next to the compound's other files. It carries
the geometry of a selected conformer (e.g. the lowest-energy one) with the
aggregated properties stamped onto it, so atom-based properties become per-atom
SDF properties that RDKit re-parses onto the atoms on load.
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rdkit import Chem

from qm_atlas.command_line.base_config import (
    CalculationInput,
    apply_software_config,
    extract_cpd_dirs,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.utils.conformer_selection import ConformerSelection, select_result_files
from qm_atlas.tasks import boltzmann
from qm_atlas.tasks.common import (
    AtomBasedProperty,
    Property,
    ScalarProperty,
    StringProperty,
    TensorProperty,
)

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas aggregate_conformer_properties"


# Maps the user-facing property_type onto the Property class used to read values.
PROPERTY_TYPE_REGISTRY: dict[str, type[Property]] = {
    "scalar": ScalarProperty,
    "tensor": TensorProperty,
    "atom_based": AtomBasedProperty,
}


# ---------------------------------------------------------------------------
# Aggregation operation configuration
# ---------------------------------------------------------------------------


class MeanAggregation(BaseModel):
    """Unweighted element-wise mean over the selected conformers."""

    operation: Literal["mean"] = Field(default="mean", description="Aggregation identifier")
    output_suffix: str = Field(
        default="_mean",
        description="Suffix appended to the property name for the aggregated value",
    )


class BoltzmannMeanAggregation(BaseModel):
    """Boltzmann-weighted element-wise mean over the selected conformers."""

    operation: Literal["boltzmann_mean"] = Field(
        default="boltzmann_mean", description="Aggregation identifier"
    )
    weight_property: str = Field(
        ...,
        description="Name of the per-conformer property holding the Boltzmann weights",
    )
    output_suffix: str = Field(
        default="_boltzmann_mean",
        description="Suffix appended to the property name for the aggregated value",
    )


class MinAggregation(BaseModel):
    """Element-wise minimum over the selected conformers."""

    operation: Literal["min"] = Field(default="min", description="Aggregation identifier")
    output_suffix: str = Field(
        default="_min",
        description="Suffix appended to the property name for the aggregated value",
    )


class MaxAggregation(BaseModel):
    """Element-wise maximum over the selected conformers."""

    operation: Literal["max"] = Field(default="max", description="Aggregation identifier")
    output_suffix: str = Field(
        default="_max",
        description="Suffix appended to the property name for the aggregated value",
    )


class StdAggregation(BaseModel):
    """Unweighted element-wise (population) standard deviation over the selected conformers."""

    operation: Literal["std"] = Field(default="std", description="Aggregation identifier")
    output_suffix: str = Field(
        default="_std",
        description="Suffix appended to the property name for the aggregated value",
    )


class BoltzmannStdAggregation(BaseModel):
    """Boltzmann-weighted element-wise (population) standard deviation over the selected conformers."""

    operation: Literal["boltzmann_std"] = Field(
        default="boltzmann_std", description="Aggregation identifier"
    )
    weight_property: str = Field(
        ...,
        description="Name of the per-conformer property holding the Boltzmann weights",
    )
    output_suffix: str = Field(
        default="_boltzmann_std",
        description="Suffix appended to the property name for the aggregated value",
    )


AggregationOpConfig = Annotated[
    MeanAggregation
    | BoltzmannMeanAggregation
    | MinAggregation
    | MaxAggregation
    | StdAggregation
    | BoltzmannStdAggregation,
    Field(discriminator="operation"),
]

OPERATION_REGISTRY = {
    "mean": MeanAggregation,
    "boltzmann_mean": BoltzmannMeanAggregation,
    "min": MinAggregation,
    "max": MaxAggregation,
    "std": StdAggregation,
    "boltzmann_std": BoltzmannStdAggregation,
}


def _deserialize_operation(v: Any) -> Any:
    """Convert a dict into the appropriate aggregation-operation config."""
    if isinstance(v, dict):
        operation_name = v.get("operation", "mean")
        operation_cls = OPERATION_REGISTRY.get(operation_name)
        if operation_cls is None:
            raise ValueError(
                f"Unknown aggregation operation: {operation_name}. "
                f"Available operations: {list(OPERATION_REGISTRY.keys())}"
            )
        return operation_cls(**v)
    return v


class PropertyAggregationSpec(BaseModel):
    """Describes how to aggregate a single named property over conformers."""

    property_name: str = Field(
        ..., description="Name of the per-conformer property to aggregate (the SDF tag)"
    )
    property_type: Literal["scalar", "tensor", "atom_based"] = Field(
        default="scalar",
        description="Expected type of the per-conformer property",
    )
    operations: list[SerializeAsAny[AggregationOpConfig]] = Field(
        default_factory=lambda: [MeanAggregation()],
        description="Aggregation operations to apply to this property",
    )

    @field_validator("operations", mode="before")
    @classmethod
    def deserialize_operations(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_deserialize_operation(item) for item in v]
        return v


# ---------------------------------------------------------------------------
# Top-level aggregation options
# ---------------------------------------------------------------------------


class RepresentativeStructureOptions(BaseModel):
    """Options for emitting a single representative structure per state.

    When enabled, one conformer per state is selected via ``geometry_selection``
    (e.g. ``keep_lowest: 1`` on an energy property) and written to
    ``{state}{output_suffix}.sdf`` in the compound directory. The aggregated
    properties are stamped onto that structure, so atom-based properties become
    per-atom SDF properties that RDKit re-parses onto the atoms on load. The file
    is not registered as a result conformer, so it never feeds back into
    selection or aggregation.
    """

    enabled: bool = Field(
        default=False,
        description="Emit a representative structure per state (off by default)",
    )
    geometry_selection: ConformerSelection = Field(
        default_factory=ConformerSelection,
        description="Selects the single conformer whose geometry represents the state "
        "(e.g. keep_lowest: 1 on an energy property). If more than one conformer is "
        "selected, the first is used",
    )
    output_suffix: str = Field(
        default="_representative",
        description="Suffix appended to the state name for the representative file and _Name",
    )


class AggregateConformerPropertiesOptions(BaseModel):
    """Options describing which properties to aggregate and how."""

    selection: ConformerSelection = Field(
        default_factory=ConformerSelection,
        description="Which conformers to aggregate over",
    )
    properties: list[PropertyAggregationSpec] = Field(
        default_factory=list,
        description="Per-property aggregation specifications",
    )
    representative_structure: RepresentativeStructureOptions = Field(
        default_factory=RepresentativeStructureOptions,
        description="Optionally emit one representative structure per state, carrying "
        "the aggregated properties on the geometry of a selected conformer",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class AggregateConformerPropertiesInterfaceConfig(BaseModel):
    """Complete configuration for the conformer-property aggregation workflow."""

    command: Literal["aggregate_conformer_properties"] = Field(
        default="aggregate_conformer_properties",
        description="CLI command name",
    )
    input: CalculationInput = Field(
        description="Results directory containing compound subdirectories to process",
    )
    aggregate_options: AggregateConformerPropertiesOptions = Field(
        default_factory=AggregateConformerPropertiesOptions,
        description="Conformer-property aggregation options",
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


def load_config(argv: list[str] | None = None) -> AggregateConformerPropertiesInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        AggregateConformerPropertiesInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Aggregate Conformer Properties Workflow — Configure via YAML or CLI arguments.\n"
        "Use 'qm-atlas conformer_properties --print-output-names --config <file>' (or the "
        "cosmotherm_properties equivalent) to look up the property tags to aggregate.",
        env_prefix="qm_atlas_AGGREGATE_CONFORMER_PROPERTIES_",
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    parser.add_class_arguments(AggregateConformerPropertiesInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = AggregateConformerPropertiesInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def _read_conformer_values(
    mols: list[Chem.Mol],
    property_name: str,
    property_cls: type[Property],
) -> list[tuple[int, np.ndarray]]:
    """Read a property from every conformer mol that carries it.

    Args:
        mols: Conformer molecules (order preserved).
        property_name: Name of the property tag to read.
        property_cls: Property class used to interpret the stored value.

    Returns:
        list[tuple[int, np.ndarray]]: ``(index, value)`` pairs for the conformers
            that carry the property, where ``index`` is the position in *mols*.
    """
    values: list[tuple[int, np.ndarray]] = []
    for idx, mol in enumerate(mols):
        try:
            prop = property_cls.from_mol(mol, property_name)
        except (KeyError, ValueError):
            continue
        values.append((idx, np.asarray(prop.get_property_value(), dtype=float)))
    return values


def _stack_values(values: list[np.ndarray]) -> np.ndarray | None:
    """Stack per-conformer values, or return None if their shapes are inconsistent."""
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        return None
    return np.stack(values, axis=0)


def _to_output_property(aggregated: np.ndarray, property_type: str) -> Property:
    """Wrap an aggregated array in a Property suitable for the molecule CSV.

    Scalars are stored as a plain float; tensor and atom-based aggregates are
    JSON-serialised into a string cell.
    """
    if property_type == "scalar":
        return ScalarProperty(float(aggregated))
    return StringProperty(json.dumps(aggregated.tolist()))


def _to_structure_property(aggregated: np.ndarray, property_type: str) -> Property:
    """Wrap an aggregated array in a Property suitable for stamping onto an SDF.

    Scalars become molecule-level properties, atom-based aggregates become
    per-atom properties, and tensors are JSON-serialised into a molecule property.
    """
    arr = np.asarray(aggregated)
    if property_type == "scalar":
        return ScalarProperty(float(arr))
    if property_type == "atom_based":
        return AtomBasedProperty(arr.tolist())
    return TensorProperty(arr)


def _aggregate_spec(
    mols: list[Chem.Mol],
    spec: PropertyAggregationSpec,
) -> dict[str, np.ndarray]:
    """Compute all configured aggregations for a single property spec.

    Returns a mapping of output property name to the aggregated array (a
    zero-dimensional array for scalars).
    """
    property_cls = PROPERTY_TYPE_REGISTRY[spec.property_type]
    indexed_values = _read_conformer_values(mols, spec.property_name, property_cls)

    if not indexed_values:
        _logger.warning(
            f"Property '{spec.property_name}' not found on any selected conformer; skipping"
        )
        return {}

    stacked = _stack_values([value for _, value in indexed_values])
    if stacked is None:
        _logger.warning(
            f"Inconsistent shapes for property '{spec.property_name}' across conformers; skipping"
        )
        return {}

    outputs: dict[str, np.ndarray] = {}
    for operation in spec.operations:
        output_name = f"{spec.property_name}{operation.output_suffix}"

        if isinstance(operation, MeanAggregation):
            aggregated = boltzmann.plain_average(stacked, axis=0)
        elif isinstance(operation, MinAggregation):
            aggregated = np.min(stacked, axis=0)
        elif isinstance(operation, MaxAggregation):
            aggregated = np.max(stacked, axis=0)
        elif isinstance(operation, StdAggregation):
            aggregated = np.std(stacked, axis=0)
        elif isinstance(operation, BoltzmannMeanAggregation):
            weights = _read_weights(mols, indexed_values, operation.weight_property)
            if weights is None:
                continue
            aggregated = boltzmann.weighted_average(stacked, weights, axis=0)
        elif isinstance(operation, BoltzmannStdAggregation):
            weights = _read_weights(mols, indexed_values, operation.weight_property)
            if weights is None:
                continue
            weighted_mean = boltzmann.weighted_average(stacked, weights, axis=0)
            weighted_variance = boltzmann.weighted_average(
                (stacked - weighted_mean) ** 2, weights, axis=0
            )
            aggregated = np.sqrt(weighted_variance)
        else:  # pragma: no cover - guarded by the discriminated union
            _logger.warning(f"Unknown aggregation operation for '{output_name}'; skipping")
            continue

        outputs[output_name] = np.asarray(aggregated)

    return outputs


def _read_weights(
    mols: list[Chem.Mol],
    indexed_values: list[tuple[int, np.ndarray]],
    weight_property: str,
) -> list[float] | None:
    """Read Boltzmann weights aligned to the conformers that carry the value.

    Returns None (skipping the operation) if any contributing conformer lacks a
    valid weight.
    """
    weights: list[float] = []
    for idx, _ in indexed_values:
        try:
            weights.append(
                ScalarProperty.from_mol(mols[idx], weight_property).get_property_value()
            )
        except (KeyError, ValueError):
            _logger.warning(
                f"Weight property '{weight_property}' missing on a selected conformer; "
                f"skipping Boltzmann aggregation"
            )
            return None
    return weights


def run_job(
    cpd_dir: compound_dir.CpdDir,
    options: AggregateConformerPropertiesOptions,
) -> None:
    """Aggregate conformer properties for every state of a compound directory."""
    if not options.properties:
        _logger.warning("No properties to aggregate; nothing to do")
        return

    for state_name in cpd_dir.registry_handler.get_registered_names():
        selected_files = select_result_files(cpd_dir, state_name, options.selection)
        if not selected_files:
            _logger.warning(f"No conformers selected for state '{state_name}', skipping")
            continue

        mols = [cpd_dir.extract_mol(sdf_file) for sdf_file in selected_files]

        csv_properties: dict[str, Property] = {}
        aggregated_arrays: dict[str, tuple[np.ndarray, str]] = {}
        for spec in options.properties:
            for output_name, aggregated in _aggregate_spec(mols, spec).items():
                csv_properties[output_name] = _to_output_property(aggregated, spec.property_type)
                aggregated_arrays[output_name] = (aggregated, spec.property_type)

        if csv_properties:
            cpd_dir.add_properties_to_molecule_csv(state_name, csv_properties)
            _logger.info(
                f"Wrote {len(csv_properties)} aggregated properties for state '{state_name}'"
            )

        if options.representative_structure.enabled and aggregated_arrays:
            _write_representative_structure(
                cpd_dir, state_name, options.representative_structure, aggregated_arrays
            )


def _write_representative_structure(
    cpd_dir: compound_dir.CpdDir,
    state_name: str,
    options: RepresentativeStructureOptions,
    aggregated_arrays: dict[str, tuple[np.ndarray, str]],
) -> None:
    """Write a representative structure SDF carrying the aggregated properties.

    The representative geometry is the conformer selected by
    ``options.geometry_selection``; the aggregated properties are stamped onto it.
    """
    geometry_files = select_result_files(cpd_dir, state_name, options.geometry_selection)
    if not geometry_files:
        _logger.warning(
            f"No conformer selected for the representative geometry of state '{state_name}'; "
            f"skipping representative structure"
        )
        return
    if len(geometry_files) > 1:
        _logger.warning(
            f"Representative geometry selection returned {len(geometry_files)} conformers for "
            f"state '{state_name}'; using '{geometry_files[0].stem}'"
        )

    donor_file = geometry_files[0]
    donor_mol = cpd_dir.extract_mol(donor_file)
    n_atoms = donor_mol.GetNumAtoms()

    property_dict: dict[str, Property] = {
        "Representative_Geometry_Source": StringProperty(donor_file.stem),
    }
    for output_name, (aggregated, property_type) in aggregated_arrays.items():
        if property_type == "atom_based" and np.asarray(aggregated).shape[0] != n_atoms:
            _logger.warning(
                f"Atom-based property '{output_name}' has {np.asarray(aggregated).shape[0]} "
                f"values but the representative geometry of state '{state_name}' has {n_atoms} "
                f"atoms; skipping it on the representative structure"
            )
            continue
        property_dict[output_name] = _to_structure_property(aggregated, property_type)

    sdf_file = cpd_dir.add_representative_structure(
        state_name, donor_mol, property_dict, options.output_suffix
    )
    _logger.info(f"Wrote representative structure for state '{state_name}' to '{sdf_file.name}'")


def run_local(config: AggregateConformerPropertiesInterfaceConfig) -> None:
    """Run the conformer-property aggregation workflow locally.

    Args:
        config: AggregateConformerPropertiesInterfaceConfig with all settings.
    """
    cpd_dirs = extract_cpd_dirs(config.input)
    if not cpd_dirs:
        _logger.warning("No compound directories found to process")
        return

    try:
        for cpd_dir in cpd_dirs:
            cpd_dir.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = cpd_dir.log_dir / "aggregate_conformer_properties.log"
            task_files.change_log_file(
                log_file,
                filemode="a",
                level=logging.DEBUG if config.verbose else logging.INFO,
                format="%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            start_time = time.perf_counter()
            try:
                _logger.info(f"Aggregating conformer properties for {cpd_dir.cpd_name}")
                run_job(cpd_dir, config.aggregate_options)
            except KeyboardInterrupt:
                _logger.error("Got ^C while aggregating conformer properties.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Aggregating conformer properties failed for {cpd_dir.cpd_name}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())
            finally:
                eval_time = time.perf_counter() - start_time
                _logger.info(f"Aggregated conformer properties in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: AggregateConformerPropertiesInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the aggregate_conformer_properties CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (AggregateConformerPropertiesInterfaceConfig | None):
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
