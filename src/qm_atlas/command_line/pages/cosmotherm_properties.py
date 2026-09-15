"""COSMOtherm property calculation CLI.

Perform COSMOtherm calculations to compute properties like logP, free energy of solvation, conformer-dependent descriptors, or PSA.
"""

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
from qm_atlas.command_line.utils.check_results import check_cosmotherm_calculation
from qm_atlas.command_line.utils.output_names import (
    PRINT_OUTPUT_NAMES_FLAGS,
    extract_config_path,
    format_output_names,
    load_options_section,
)
from qm_atlas.tasks import cosmo_properties

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas cosmotherm_properties"


OPTIONS_REGISTRY = {
    "cosmo_logp": cosmo_properties.CosmoLogPConfig,
    "cosmo_delta_g": cosmo_properties.CosmoDeltaGConfig,
    "cosmo_descriptors": cosmo_properties.CosmoDescriptorsConfig,
    "cosmo_psa": cosmo_properties.CosmoPsaConfig,
    "cosmo_perm": cosmo_properties.CosmoPermConfig,
}


class CosmoPropertyOptions(BaseModel):

    calculation_tasks: list[cosmo_properties.CosmoTaskConfig] = Field(
        default_factory=list, description="List of COSMO calculation tasks to perform."
    )

    @staticmethod
    def _deserialize_config(v: Any) -> Any:
        """Convert dict to appropriate COSMO options class based on backend field."""
        if isinstance(v, dict):
            config_name = v.get("backend", "undefined")
            if config_name not in OPTIONS_REGISTRY:
                raise ValueError(f"Unknown COSMO calculation name: {config_name}")
            options_cls = OPTIONS_REGISTRY[config_name]
            return options_cls(**v)
        return v

    @field_validator("calculation_tasks", mode="before")
    @classmethod
    def deserialize_configs(cls, v: Any) -> Any:
        """Deserialize COSMO config sequences from dicts."""
        if isinstance(v, list):
            return [cls._deserialize_config(item) for item in v]
        return v

    @model_validator(mode="after")
    def validate_calculation_tasks(self) -> "CosmoPropertyOptions":
        """Validate that there will be no clashes of property names between different calculation tasks."""
        tasks_to_calculation_prefices = defaultdict(set)
        for task in self.calculation_tasks:
            calculator_name = task.backend
            property_prefix = task.get_property_prefix()
            if property_prefix in tasks_to_calculation_prefices[calculator_name]:
                raise ValueError(
                    f"You have several instances of {calculator_name} calculations with the same property prefix {property_prefix}."
                    "This will lead to clashes in property names. Please make sure that each calculation task has a unique property prefix."
                )
            tasks_to_calculation_prefices[calculator_name].add(property_prefix)
        return self


# ---------------------------------------------------------------------------
# Different Defaults for Submitted Jobs
# ---------------------------------------------------------------------------


class CosmoPropertySubmissionConfig(SubmissionConfig):
    # Inherit all fields from SubmissionConfig, override certain default values for this workflow
    cores_per_task: int = Field(
        default=1,
        description="Number of cores to allocate per task",
        ge=1,
    )
    max_time: str = Field(
        default="4hours",
        description="Maximum wall-clock time per job (e.g., '10hours', '2days')",
    )
    name: str = Field(
        default="QMA_CosmoProp",
        description="Name for the job",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class CosmoPropertyInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for COSMOtherm Property workflow."""

    command: Literal["cosmotherm_properties"] = Field(
        default="cosmotherm_properties",
        description="CLI command name",
    )
    submission_config: CosmoPropertySubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=CosmoPropertySubmissionConfig,
        description="UGE job submission settings",
    )
    cosmo_property_options: CosmoPropertyOptions = Field(
        default_factory=CosmoPropertyOptions,
        description="COSMOtherm Property workflow specific options",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> CosmoPropertyInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        CosmoPropertyInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="COSMOtherm Property Workflow — Configure via YAML or CLI arguments.\n"
        "Run 'qm-atlas describe cosmo_tasks' to see available COSMO task types and their options.",
        env_prefix="qm_atlas_COSMO_PROPERTY_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full CosmoPropertyInterfaceConfig as structured arguments
    parser.add_class_arguments(CosmoPropertyInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = CosmoPropertyInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_local(config: CosmoPropertyInterfaceConfig) -> None:
    """Run COSMOtherm Property workflow locally.

    Args:
        config: CosmoPropertyInterfaceConfig with all settings.
    """

    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="cosmotherm_properties")

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
                _logger.error("Got ^C while running COSMOtherm property jobs.")
                sys.exit()

            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"COSMOtherm Property workflow failed for {sdf_file}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())

            finally:
                stop_time = time.perf_counter()
                eval_time = stop_time - start_time
                _logger.info(f"Evaluated COSMOtherm Property workflow in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def run_job(
    cpd_dir: compound_dir.CpdDir,
    sdf_file: Path,
    config: CosmoPropertyInterfaceConfig,
) -> None:

    _logger.info(f"Working on Molecule {sdf_file.stem}")
    cosmo_property_options = config.cosmo_property_options

    cpd_name = cpd_dir.find_state_by_file(sdf_file)
    sdf_files = cpd_dir.get_result_files(cpd_name)

    cosmo_files = list()
    cosmo_idx_to_sdf_idx = list()
    for idx, sdf_file in enumerate(sdf_files):
        cosmo_file = cpd_dir.get_associated_cosmo_file(sdf_file)
        if not cosmo_file.exists():
            _logger.warning(f"Could not find COSMO file for {sdf_file}, skipping")
            continue
        cosmo_files.append(cosmo_file)
        cosmo_idx_to_sdf_idx.append(idx)

    # Perform COSMO property calculations
    for task_options in cosmo_property_options.calculation_tasks:
        _logger.info(f"Running COSMO calculation task: {task_options.backend}")

        try:
            calculated_results = cosmo_properties.run_cosmo_calculation(
                conformer_cosmo_files=cosmo_files,
                config=task_options,
            )
        except RuntimeError as exc:
            _logger.error(f"COSMO calculation task {task_options.backend} failed: {exc}")
            continue

        if calculated_results is None:
            _logger.error(f"COSMO calculation task {task_options.backend} returned no results.")
            continue

        if isinstance(calculated_results, dict):
            # Single result (e.g., logP, delta G)
            cpd_dir.add_properties_to_molecule_csv(cpd_name, calculated_results)

        elif isinstance(calculated_results, list):
            # Multiple results (e.g., descriptors for multiple conformers)
            for idx, result in enumerate(calculated_results):
                sdf_idx = cosmo_idx_to_sdf_idx[idx]
                cpd_dir.add_properties_to_sdf(sdf_files[sdf_idx], result)
                cpd_dir.add_properties_to_conformer_csv(sdf_files[sdf_idx], result)

        _logger.info(f"Finished COSMO calculation task: {task_options.backend}")


# ---------------------------------------------------------------------------
# Job Submission
# ---------------------------------------------------------------------------


def submit(config: CosmoPropertyInterfaceConfig) -> None:
    """Submit COSMOtherm Property jobs to cluster."""
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

    # Extract worker paths (molecules to process)
    worker_paths = extract_worker_paths(config.input, workflow_type="cosmotherm_properties")

    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} molecules to process")

    submission.submit(
        worker_paths=worker_paths,
        config=config,
        command=COMMAND,
    )

    if config.submission_config.wait:
        check_cosmotherm_calculation(worker_paths, config.cosmo_property_options.calculation_tasks)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def print_output_names(argv: list[str]) -> None:
    """Print the SDF property tags produced by the configured COSMO tasks.

    Reads the ``cosmo_property_options`` section from the ``--config`` file and
    lists the output property names without running any calculation.
    """
    config_path = extract_config_path(argv)
    if config_path is None:
        raise SystemExit("--print-output-names requires a --config <file> argument")
    section = load_options_section(config_path, "cosmo_property_options")
    options = CosmoPropertyOptions(**section)
    print(format_output_names(options.calculation_tasks))


def main(
    argv: list[str] | None = None,
    config: CosmoPropertyInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the COSMOtherm Property CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (CosmoPropertyInterfaceConfig | None):
            Optional CosmoPropertyInterfaceConfig object. If provided, overrides command-line arguments.
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
