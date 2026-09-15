import logging
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, field_validator, model_validator

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.utils import submission
from qm_atlas.command_line.utils.check_results import check_conformer_property_calculation
from qm_atlas.command_line.utils.conformer_selection import ConformerSelection, filter_worker_paths
from qm_atlas.command_line.utils.output_names import (
    PRINT_OUTPUT_NAMES_FLAGS,
    extract_config_path,
    format_output_names,
    load_options_section,
)
from qm_atlas.constants import COSMO_OUTPUT_KEY
from qm_atlas.tasks import calculate_properties

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas conformer_properties"


OPTIONS_REGISTRY = {
    calculate_properties.XTB_SP_KEY: calculate_properties.XtbSinglePointOptions,
    calculate_properties.TM_SP_KEY: calculate_properties.TurbomoleSinglePointOptions,
    calculate_properties.TM_FH_KEY: calculate_properties.TurbomoleFreehOptions,
    calculate_properties.TM_FUKUI_KEY: calculate_properties.TurbomoleFukuiOptions,
    calculate_properties.TM_NMR_KEY: calculate_properties.TurbomoleNmrShieldingOptions,
    calculate_properties.TM_VCD_KEY: calculate_properties.TurbomoleVcdOptions,
    calculate_properties.JAGUAR_HABS_KEY: calculate_properties.JaguarHydrogenAbstractionOptions,
    calculate_properties.JAGUAR_PROPS_KEY: calculate_properties.JaguarQmDescriptorsOptions,
}

# TODO: how to control when a cosmo file is written to the results directory


class ConformerPropertyOptions(BaseModel):

    calculation_tasks: list[calculate_properties.CalculationConfig] = Field(
        default_factory=list, description="List of calculation tasks to perform on the conformer."
    )

    @staticmethod
    def _deserialize_config(v: Any) -> Any:
        """Convert dict to appropriate options class based on the backend field."""
        # jsonargparse may pre-instantiate items as the base class before Pydantic's
        # field_validator runs; convert back to dict so we can re-dispatch correctly.
        if isinstance(v, calculate_properties.PropertyCalculatorOptions):
            v = v.model_dump()
        if isinstance(v, dict):
            backend = v.get("backend", "undefined")
            if backend not in OPTIONS_REGISTRY:
                raise ValueError(f"Unknown calculator name: {backend}")
            options_cls = OPTIONS_REGISTRY[backend]
            return options_cls(**v)
        return v

    @field_validator("calculation_tasks", mode="before")
    @classmethod
    def deserialize_configs(cls, v: Any) -> Any:
        """Deserialize optimization config sequences from dicts."""
        if isinstance(v, list):
            return [cls._deserialize_config(item) for item in v]
        return v

    @model_validator(mode="after")
    def validate_calculation_tasks(self) -> "ConformerPropertyOptions":
        """Validate that there will be no clashes of property names between different calculation tasks."""

        tasks_to_calculation_prefices = defaultdict(set)
        for prop_options in self.calculation_tasks:
            backend = prop_options.backend
            property_prefix = prop_options.get_property_prefix()
            if property_prefix in tasks_to_calculation_prefices[backend]:
                raise ValueError(
                    f"You have several instances of {backend} calculations with the same property prefix {property_prefix}."
                    "This will lead to clashes in property names. Please make sure that each calculation task has a unique property prefix."
                )
            tasks_to_calculation_prefices[backend].add(property_prefix)

        return self


# ---------------------------------------------------------------------------
# Different Defaults for Submitted Jobs
# ---------------------------------------------------------------------------


class ConformerPropertySubmissionConfig(SubmissionConfig):
    # Inherit all fields from SubmissionConfig, override certain default values for this workflow
    cores_per_task: int = Field(
        default=1,
        description="Number of cores to allocate per task",
        ge=1,
    )
    max_time: str = Field(
        default="1day",
        description="Maximum wall-clock time per job (e.g., '10hours', '2days')",
    )
    name: str = Field(
        default="QMA_Prop",
        description="Name for the job",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class ConformerPropertyInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for Conformer Property workflow."""

    command: Literal["conformer_properties"] = Field(
        default="conformer_properties",
        description="CLI command name",
    )
    submission_config: ConformerPropertySubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=ConformerPropertySubmissionConfig,
        description="UGE job submission settings",
    )
    conformer_property_options: ConformerPropertyOptions = Field(
        default_factory=ConformerPropertyOptions,
        description="Conformer Property workflow specific options",
    )
    conformer_selection: ConformerSelection = Field(
        default_factory=ConformerSelection,
        description="Select the subset of result conformers to run the calculations on "
        "(e.g. only the lowest-energy conformer, or only reference results)",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> ConformerPropertyInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        InterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Conformer Property Workflow — Configure via YAML or CLI arguments.\n"
        "Run 'qm-atlas describe calculators' to see available property calculators and their options.",
        env_prefix="qm_atlas_CONFORMER_PROPERTY_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full ConformerPropertyInterfaceConfig as structured arguments
    parser.add_class_arguments(ConformerPropertyInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = ConformerPropertyInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_local(config: ConformerPropertyInterfaceConfig) -> None:
    """Run Conformer Property workflow locally.

    Args:
        config: ConformerPropertyInterfaceConfig with all settings.
    """

    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="conformer_properties")
    worker_paths = filter_worker_paths(worker_paths, config.conformer_selection)

    prev_log_file = None
    try:
        for cpd_dir, sdf_file, log_file in worker_paths:

            # use new log file, if required
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file

            start_time = time.perf_counter()

            try:
                run_job(cpd_dir, sdf_file, config)

            except KeyboardInterrupt:
                _logger.error("Got ^C while running conformer properties jobs.")
                sys.exit()

            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Conformer Properties workflow failed for {sdf_file}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())

            finally:
                stop_time = time.perf_counter()
                eval_time = stop_time - start_time
                _logger.info(f"Evaluated Conformer Properties workflow in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def run_job(
    cpd_dir: compound_dir.CpdDir,
    sdf_file: Path,
    config: ConformerPropertyInterfaceConfig,
) -> None:

    _logger.info(f"Working on Molecule {sdf_file.stem}")
    mol = cpd_dir.extract_mol(sdf_file)
    conformer_property_options = config.conformer_property_options

    # Perform property calculations
    for task_options in conformer_property_options.calculation_tasks:
        _logger.info(f"Running calculation task: {task_options.backend}")
        calculated_props = calculate_properties.calculate_property(
            mol=mol,
            conf_id=0,
            config=task_options,
            n_cores=config.n_cores or 1,
        )

        if calculated_props is None:
            _logger.error(f"Calculation task {task_options.backend} returned no properties.")
            continue

        if COSMO_OUTPUT_KEY in calculated_props:
            _logger.info(
                f"Adding COSMO output as cosmo file to results directory for molecule {sdf_file.stem}"
            )
            cosmo_output = calculated_props.pop(COSMO_OUTPUT_KEY).get_property_value()
            cpd_dir.add_cosmo_result(sdf_file, cosmo_output, overwrite=True)

        cpd_dir.add_properties_to_sdf(sdf_file, calculated_props)
        cpd_dir.add_properties_to_conformer_csv(sdf_file, calculated_props)
        _logger.info(f"Finished calculation task: {task_options.backend}")


# ---------------------------------------------------------------------------
# Job Submission
# ---------------------------------------------------------------------------


def submit(config: ConformerPropertyInterfaceConfig) -> None:
    """Submit Conformer Property jobs to cluster."""
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

    # Extract worker paths (molecules to process)
    worker_paths = extract_worker_paths(config.input, workflow_type="conformer_properties")
    worker_paths = filter_worker_paths(worker_paths, config.conformer_selection)

    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} conformers to process")

    submission.submit(
        worker_paths=worker_paths,
        config=config,
        command=COMMAND,
    )

    if config.submission_config.wait:
        check_conformer_property_calculation(
            worker_paths, config.conformer_property_options.calculation_tasks
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def print_output_names(argv: list[str]) -> None:
    """Print the SDF property tags produced by the configured calculation tasks.

    Reads the ``conformer_property_options`` section from the ``--config`` file
    and lists the output property names without running any calculation.
    """
    config_path = extract_config_path(argv)
    if config_path is None:
        raise SystemExit("--print-output-names requires a --config <file> argument")
    section = load_options_section(config_path, "conformer_property_options")
    options = ConformerPropertyOptions(**section)
    print(format_output_names(options.calculation_tasks))


def main(
    argv: list[str] | None = None,
    config: ConformerPropertyInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the Conformer Property CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (ConformerPropertyInterfaceConfig | None):
            Optional ConformerPropertyInterfaceConfig object. If provided, overrides command-line arguments.
        log_file (Path | None):
            Optional file to append logs to.
    """
    if config is None:
        if argv is None:
            argv = sys.argv[1:]
        if any(flag in argv for flag in PRINT_OUTPUT_NAMES_FLAGS):
            print_output_names(argv)
            return
        config = load_config(argv=argv)
    apply_software_config(config.software_config)
    setup_logging(verbose=config.verbose, log_file=log_file)

    if config.submission_config.submit:
        submit(config)
    else:
        run_local(config)


if __name__ == "__main__":
    main()
