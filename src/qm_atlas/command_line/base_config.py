"""Utilities for command-line argument handling and validation."""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir, find_cpd_dirs
from qm_atlas.command_line.file_interface.submission_files import (
    WorkerPath,
    get_task_work_items,
    read_task_control,
)
from qm_atlas.software_environment import SOFTWARE_CONFIG, SOFTWARE_CONFIG_ENV_VAR

_logger = logging.getLogger(__name__)


DEFAULT_LOG_DIR_NAME = "log"


class CalculationInput(BaseModel):
    """Configuration for accessing existing results directories or compound directories.

    Requirements:
    - results_directory is always required
    - Exactly one of compound_directories or task_control_file can be provided (mutually exclusive)
    - If task_id is provided, task_control_file must also be provided
    """

    results_directory: Path | None = Field(
        description="Results directory containing compound subdirectories",
    )
    compound_directories: list[Path] | None = Field(
        default=None,
        description="List of compound directories (mutually exclusive with task_control_file)",
    )
    task_control_file: Path | None = Field(
        default=None,
        description="Path to task control JSON file (mutually exclusive with compound_directories)",
    )
    task_id: int | None = Field(
        default=None,
        description="Task ID for array jobs (requires task_control_file to be provided)",
    )

    @field_validator("results_directory", mode="after")
    @classmethod
    def validate_results_directory_field(cls, v: Path | None) -> Path | None:
        """Validate results directory using arg_utils validation.

        For read_input, the results directory is created if it doesn't exist.
        Pydantic handles conversion to Path.
        """
        if v is not None:
            validate_results_dir(v, create_if_missing=False)
        return v.expanduser().resolve() if v is not None else None

    @field_validator("compound_directories", mode="before")
    @classmethod
    def validate_compound_directories_field(
        cls, v: list[Path | str] | str | Path
    ) -> list[Path] | None:
        """Convert compound_directories to list of Paths and validate.

        Ensures all paths exist and are directories.
        """
        if v is None:
            return None

        if isinstance(v, (str, Path)):
            v = [v]

        paths = [Path(p).expanduser().resolve() for p in v]

        for path in paths:
            if not path.exists():
                raise ValueError(f"Input path does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Input path is not a directory: {path}")
            if not compound_dir.is_cpd_dir(path):
                raise ValueError(f"Input path is not a valid compound directory: {path}")

        return paths

    @field_validator("task_control_file", mode="after")
    @classmethod
    def validate_task_control_file(cls, input_path: Path | None) -> Path | None:
        """Validate task control file if explicitly provided."""
        if input_path is None:
            return None

        path = input_path.expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Task control file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Task control file is not a file: {path}")
        if path.suffix != ".json":
            raise ValueError(f"Task control file must be JSON: {path}")
        return path

    @model_validator(mode="after")
    def validate_mutual_exclusivity(self) -> "CalculationInput":
        """Validate that compound_directories and task_control_file are mutually exclusive."""
        if self.compound_directories is not None and self.task_control_file is not None:
            raise ValueError(
                "Cannot provide both compound_directories and task_control_file. "
                "Please provide only one."
            )

        return self

    @model_validator(mode="after")
    def validate_task_id_requires_task_control_file(self) -> "CalculationInput":
        """Validate that task_id is only provided with task_control_file."""
        if self.task_id is not None and self.task_control_file is None:
            raise ValueError(
                "Cannot provide task_id without task_control_file. "
                "Please provide a task control file when using task IDs."
            )

        return self


class SubmissionConfig(BaseModel):
    """Configuration for UGE (Univa Grid Engine) job submission.

    This class is designed to be used across multiple CLIs with different defaults.
    Use the `with_defaults()` class method to create instances with CLI-specific defaults,
    or provide defaults when creating the parent config model.
    """

    submit: bool = Field(
        default=False,
        description="Whether to submit jobs to UGE cluster (if False, runs locally)",
    )
    engine: str = Field(
        default="uge",
        description="HPC job submission engine (currently only 'uge' supported)",
    )
    cores_per_task: int = Field(
        default=1,
        description="Number of CPU cores per task",
        ge=1,
    )
    mem_per_core: int = Field(
        default=4,
        description="Memory per core in GB",
    )
    jobs_per_task: int = Field(
        default=1,
        description="Number of jobs to process per task",
        ge=1,
    )
    max_total_num_cores: int = Field(
        default=100,
        description="Maximum total number of cores to allocate across all tasks",
        ge=1,
    )
    max_time: str = Field(
        default="1day",
        description="Maximum wall-clock time per job (e.g., '10hours', '2days')",
    )
    name: str = Field(
        default="uge_job",
        description="Name for the job",
    )
    user_email: str | None = Field(
        default=None,
        description="Email address to notify when jobs complete (must be valid email format)",
    )
    wait: bool = Field(
        default=False,
        description="Wait for jobs to complete before exiting",
    )

    @model_validator(mode="after")
    def validate_cores_configuration(self) -> "SubmissionConfig":
        """Ensure max_total_num_cores is greater than or equal to cores_per_task."""
        if self.max_total_num_cores < self.cores_per_task:
            raise ValueError(
                f"max_total_num_cores ({self.max_total_num_cores}) must be greater than "
                f"or equal to cores_per_task ({self.cores_per_task})"
            )
        return self


class BaseInterfaceConfig(BaseModel):
    """Base configuration with common input and submission settings."""

    input: CalculationInput = Field(
        description="Results directory containing compound subdirectories with input structures",
    )
    submission_config: SubmissionConfig = Field(
        default_factory=SubmissionConfig,
        description="UGE job submission settings",
    )
    software_config: Path | None = Field(
        default=None,
        description="Path to software configuration YAML file (overrides qm_atlas_SOFTWARE_CONFIG_FILE env var)",
    )
    n_cores: int | None = Field(
        default=None,
        description="Number of cores for local execution. Must be specified if job submission is disabled.",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )


def apply_software_config(software_config_path: Path | None) -> None:
    """Apply a software config file by setting the env var and reloading SOFTWARE_CONFIG.

    This should be called early in each CLI main() function after config loading.
    If software_config_path is provided, validates the file exists, sets the
    qm_atlas_SOFTWARE_CONFIG_FILE environment variable, and reloads the global
    SOFTWARE_CONFIG singleton.

    Args:
        software_config_path: Path to software config YAML file, or None to skip.
    """
    if software_config_path is None:
        return

    software_config_path = software_config_path.expanduser().resolve()
    if not software_config_path.is_file():
        raise FileNotFoundError(f"Software config file not found: {software_config_path}")

    os.environ[SOFTWARE_CONFIG_ENV_VAR] = str(software_config_path)
    SOFTWARE_CONFIG.reload_environment_manager(software_config_path)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging based on config verbosity level.

    Args:
        verbose: Enable verbose logging
        log_file: Optional file to append logs to.
    """
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = (
        [logging.StreamHandler()]
        if log_file is None
        else [logging.FileHandler(log_file, mode="a")]
    )

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def validate_results_dir(
    results_dir: Path,
    create_if_missing: bool = True,
):
    """Validate and prepare the results directory.

    Args:
        results_dir: Path to the results directory
        create_if_missing: If True, create the directory if it doesn't exist.
                          If False, raise an error if the directory doesn't exist.

    Raises:
        ValueError: If parent directory doesn't exist or results_dir validation fails
    """
    results_dir = Path(results_dir)

    # Check that parent directory exists
    parent_dir = results_dir.parent
    if not parent_dir.exists():
        raise ValueError(
            f"Parent directory does not exist: {parent_dir}. "
            f"Please create it first or provide a path under an existing directory."
        )

    if not parent_dir.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent_dir}")

    # Check/create the results directory itself
    if results_dir.exists():
        if not results_dir.is_dir():
            raise ValueError(f"Path exists but is not a directory: {results_dir}")
        _logger.debug(f"Results directory already exists: {results_dir}")
    else:
        if create_if_missing:
            _logger.debug(f"Creating results directory: {results_dir}")
            results_dir.mkdir(parents=False, exist_ok=True)
        else:
            raise ValueError(
                f"Results directory does not exist: {results_dir}. "
                f"Use --create-results-dir to create it automatically."
            )


def get_input_worker_paths(cpd_dir: compound_dir.CpdDir) -> list[WorkerPath]:
    """Searches for input sdf files under *cpd_dir* and returns a list of their paths.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to search under.

    Returns:
        list[WorkerPath]: A list of WorkerPath objects containing the compound directory,
        the input sdf file path, and the associated log file path.
    """
    worker_paths: list[WorkerPath] = []
    sdf_files = cpd_dir.get_input_sdf_files()
    for sdf_file in sdf_files:
        log_file = cpd_dir.get_associated_log_file(sdf_file)
        worker_paths.append((cpd_dir, sdf_file, log_file))
    return worker_paths


def get_reference_worker_paths(cpd_dir: compound_dir.CpdDir) -> list[WorkerPath]:
    """Searches for reference sdf files under *cpd_dir* and returns a list of their paths.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to search under.

    Returns:
        list[WorkerPath]: A list of WorkerPath objects containing the compound directory,
        the reference sdf file path, and the associated log file path.
    """
    worker_paths: list[WorkerPath] = []
    for name in cpd_dir.registry_handler.get_registered_names():
        sdf_files = cpd_dir.get_reference_inputs(name)
        for sdf_file in sdf_files:
            log_file = cpd_dir.get_associated_log_file(sdf_file)
            worker_paths.append((cpd_dir, sdf_file, log_file))
    return worker_paths


def get_conformer_property_worker_paths(cpd_dir: compound_dir.CpdDir) -> list[WorkerPath]:
    """Searches for result sdf files under *cpd_dir* and returns a list of their paths.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to search under.

    Returns:
        list[WorkerPath]: A list of WorkerPath objects containing the compound directory,
        the result sdf file path, and the associated log file path.
    """
    worker_paths: list[WorkerPath] = []
    for name in cpd_dir.registry_handler.get_registered_names():
        sdf_files = cpd_dir.get_result_files(name)
        for sdf_file in sdf_files:
            log_file = cpd_dir.get_associated_log_file(sdf_file)
            worker_paths.append((cpd_dir, sdf_file, log_file))
    return worker_paths


def get_cosmo_property_worker_paths(cpd_dir: compound_dir.CpdDir) -> list[WorkerPath]:
    """Searches for conformer COSMO files under *cpd_dir* and returns a list of their paths.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to search under.

    Returns:
        list[WorkerPath]: A list of WorkerPath objects containing the compound directory,
        the COSMO file path, and the associated log file path.
    """
    worker_paths: list[WorkerPath] = []
    for name in cpd_dir.registry_handler.get_registered_names():
        sdf_files = cpd_dir.get_result_files(name)
        if len(sdf_files) == 0:
            _logger.warning(f"No result sdf files found for {name} in {cpd_dir}")
            continue
        sdf_file = sdf_files[0]  # Use first sdf file for whole-molecule jobs
        log_file = cpd_dir.get_associated_log_file(sdf_file)
        worker_paths.append((cpd_dir, sdf_file, log_file))

    return worker_paths


def get_solvent_screening_worker_paths(cpd_dir: compound_dir.CpdDir) -> list[WorkerPath]:
    """Returns a single worker path per compound directory for solvent screening.

    Solvent screening collects all neutral-state COSMO conformers in one call,
    so only one entry per compound directory is needed.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to search under.

    Returns:
        list[WorkerPath]: A list containing at most one WorkerPath for the compound.
    """
    cosmo_paths = get_cosmo_property_worker_paths(cpd_dir)
    if not cosmo_paths:
        return []
    return [cosmo_paths[0]]


FUNC_FROM_WORKFLOW_TYPE = {
    "conformer_expansion": get_input_worker_paths,
    "reference_optimization": get_reference_worker_paths,
    "conformer_properties": get_conformer_property_worker_paths,
    "cosmotherm_properties": get_cosmo_property_worker_paths,
    "solvent_screening": get_solvent_screening_worker_paths,
    "cocrystal_screening": get_solvent_screening_worker_paths,
}


def _conformer_expansion_completed(cpd_dir: compound_dir.CpdDir, sdf_file: Path) -> bool:
    state = cpd_dir.find_state_by_file(sdf_file)
    # Reference-optimization results share the results/ dir, so exclude them here.
    return len(cpd_dir.get_result_files(state, include_references=False)) > 0


def _reference_optimization_completed(cpd_dir: compound_dir.CpdDir, sdf_file: Path) -> bool:
    state = cpd_dir.find_state_by_file(sdf_file)
    return len(cpd_dir.get_reference_results(state, sdf_file)) > 0


# Predicate deciding whether a worker path already has results and can be skipped.
WORKER_PATH_COMPLETE_FROM_WORKFLOW = {
    "conformer_expansion": _conformer_expansion_completed,
    "reference_optimization": _reference_optimization_completed,
}


def worker_path_completed(
    cpd_dir: compound_dir.CpdDir, sdf_file: Path, workflow_type: str
) -> bool:
    """Return whether *sdf_file* already has results for *workflow_type*.

    Workflows without a registered completeness check are never considered done.

    Args:
        cpd_dir (compound_dir.CpdDir): The owning compound directory.
        sdf_file (Path): The worker-path input file to check.
        workflow_type (str): The workflow whose results define completeness.

    Returns:
        bool: True if results already exist and the worker path can be skipped.
    """
    checker = WORKER_PATH_COMPLETE_FROM_WORKFLOW.get(workflow_type)
    if checker is None:
        return False
    return checker(cpd_dir, sdf_file)


def extract_worker_paths(
    calculation_input: CalculationInput,
    workflow_type: str,
    skip_completed: bool = False,
) -> list[WorkerPath]:
    """Extract worker paths based on input configuration.

    Args:
        calculation_input (CalculationInput): The resolved input configuration.
        workflow_type (str): The workflow whose worker paths are being built.
        skip_completed (bool): When True, drop worker paths whose results already
            exist (used to resume a partially completed run). Only applied to the
            compound-directory and results-directory branches; task-control slices
            are assumed to have been filtered when the task array was created.

    Returns:
        list[WorkerPath]: The worker paths to process.
    """
    worker_paths: list[WorkerPath] = []

    # Case 1: Extract from task control file
    if calculation_input.task_control_file is not None:

        # If task_id is specified, extract only the relevant paths for that task
        if calculation_input.task_id is not None:
            return get_task_work_items(
                calculation_input.task_control_file, calculation_input.task_id
            )

        # extract all worker paths if no specific task_id is provided
        task_control_dict = read_task_control(calculation_input.task_control_file)
        for w_tasks in task_control_dict.values():
            worker_paths.extend(w_tasks)
        return worker_paths

    func = FUNC_FROM_WORKFLOW_TYPE.get(workflow_type)
    if func is None:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")

    # Case 2: Extract from compound directories
    if calculation_input.compound_directories is not None:
        for cpd_path in calculation_input.compound_directories:
            worker_paths.extend(func(create_cpd_dir(cpd_path)))

    # Case 3: Extract from results directory
    else:
        cpd_dirs = find_cpd_dirs(calculation_input.results_directory)
        for cpd_dir in cpd_dirs:
            worker_paths.extend(func(cpd_dir))

    if skip_completed:
        worker_paths = [
            (cpd_dir, sdf_file, log_file)
            for cpd_dir, sdf_file, log_file in worker_paths
            if not worker_path_completed(cpd_dir, sdf_file, workflow_type)
        ]

    return worker_paths


def extract_cpd_dirs(calculation_input: CalculationInput) -> list[compound_dir.CpdDir]:
    """Return the compound directories referenced by *calculation_input*.

    Unlike :func:`extract_worker_paths`, this resolves whole compound directories
    rather than per-conformer worker paths, which is what post-processing commands
    (collect, averaging, Boltzmann weights) operate on.

    Args:
        calculation_input (CalculationInput): The resolved input configuration.

    Returns:
        list[compound_dir.CpdDir]: The unique compound directories to process.
    """
    if calculation_input.compound_directories is not None:
        return [create_cpd_dir(path) for path in calculation_input.compound_directories]

    if calculation_input.task_control_file is not None:
        task_control_dict = read_task_control(calculation_input.task_control_file)
        seen: dict[str, compound_dir.CpdDir] = {}
        for worker_paths in task_control_dict.values():
            for cpd_dir, _sdf_file, _log_file in worker_paths:
                seen.setdefault(str(cpd_dir.dir.resolve()), cpd_dir)
        return list(seen.values())

    return find_cpd_dirs(calculation_input.results_directory)
