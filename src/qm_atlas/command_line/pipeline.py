"""Pipeline CLI for running multiple qm_atlas commands in sequence.

Allows combining multiple CLI steps into a single pipeline run,
configured via a YAML file. Paths for input files and output
(results directory) can be provided as CLI arguments instead of being
hard-coded in the config.

Pipeline Behavior:
    - When a step has ``submit: true`` in its ``submission_config``, the
      pipeline automatically forces ``wait: true`` so that subsequent steps
      only begin after the submitted jobs have finished.
    - The pipeline itself can be submitted to the cluster using the top-level
      ``submission_config.submit: true`` option.  In that case, the resolved
      config is dumped to a YAML file and the pipeline is re-invoked on the
      cluster with ``qm-atlas pipeline --config <resolved_config>.yaml``.

Example YAML config::

    steps:
      - command: read_input

      - command: fast_conformers
        submission_config:
          submit: true

      - command: add_reference_conformers

      - command: optimize_reference_conformers
        submission_config:
          submit: true

      - command: conformer_properties
        submission_config:
          submit: true
        conformer_property_options:
          calculation_tasks:
            - backend: "turbomole_single_point"

      - command: cosmotherm_properties
        submission_config:
          submit: true

Usage::

    qm-atlas pipeline --config pipeline.yaml \\
        --input_file ./molecules.sdf \\
        --results_directory ./results

    # With separate reference conformer input files:
    qm-atlas pipeline --config pipeline.yaml \\
        --input_file ./molecules.sdf \\
        --reference_input_file ./ref_conformers.sdf \\
        --results_directory ./results

    # Submit the pipeline itself to the cluster:
    qm-atlas pipeline --config pipeline.yaml \\
        --input_file ./molecules.sdf \\
        --results_directory ./results \\
        --submission_config.submit true
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml
from hpc_funcs.files import generate_name
from hpc_funcs.schedulers.uge.qsub import write_script
from hpc_funcs.schedulers.uge.submission import generate_script
from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, field_validator, model_validator

from qm_atlas.command_line.base_config import (
    SubmissionConfig,
    apply_software_config,
    setup_logging,
    validate_results_dir,
)
from qm_atlas.command_line.file_interface import task_files
from qm_atlas.command_line.file_interface.submission_files import get_submission_scratch_dir
from qm_atlas.command_line.pages import (
    add_reference_conformers,
    aggregate_conformer_properties,
    align,
    calculate_boltzmann_weights,
    cocrystal_screening,
    collect,
    conformer_properties,
    cosmo_pka,
    cosmotherm_properties,
    fast_conformers,
    generate_tautomers,
    optimize_reference_conformers,
    protonate,
    read_input,
    rescoss_conformers,
    solvent_screening,
)
from qm_atlas.command_line.utils.submission import (
    parse_time,
    submit_script_with_retry,
    wait_for_jobs_using_hold_job,
)
from qm_atlas.software_environment import SOFTWARE_CONFIG, SOFTWARE_CONFIG_ENV_VAR

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas pipeline"

# Registry mapping command names to (main_function, config_class) tuples
STEP_REGISTRY: dict[str, tuple[Any, type[BaseModel]]] = {
    "read_input": (read_input.main, read_input.ReadInputConfig),
    "fast_conformers": (fast_conformers.main, fast_conformers.FastConformersInterfaceConfig),
    "add_reference_conformers": (
        add_reference_conformers.main,
        add_reference_conformers.ReferenceConformersInputConfig,
    ),
    "rescoss_conformers": (rescoss_conformers.main, rescoss_conformers.RescossInterfaceConfig),
    "optimize_reference_conformers": (
        optimize_reference_conformers.main,
        optimize_reference_conformers.RefInterfaceConfig,
    ),
    "conformer_properties": (
        conformer_properties.main,
        conformer_properties.ConformerPropertyInterfaceConfig,
    ),
    "cosmotherm_properties": (
        cosmotherm_properties.main,
        cosmotherm_properties.CosmoPropertyInterfaceConfig,
    ),
    "generate_tautomers": (
        generate_tautomers.main,
        generate_tautomers.GenerateTautomersInterfaceConfig,
    ),
    "protonate": (protonate.main, protonate.ProtonateInterfaceConfig),
    "cosmo_pka": (cosmo_pka.main, cosmo_pka.CosmoPkaInterfaceConfig),
    "solvent_screening": (
        solvent_screening.main,
        solvent_screening.SolventScreeningInterfaceConfig,
    ),
    "cocrystal_screening": (
        cocrystal_screening.main,
        cocrystal_screening.CocrystalScreeningInterfaceConfig,
    ),
    "calculate_boltzmann_weights": (
        calculate_boltzmann_weights.main,
        calculate_boltzmann_weights.CalculateBoltzmannWeightsInterfaceConfig,
    ),
    "aggregate_conformer_properties": (
        aggregate_conformer_properties.main,
        aggregate_conformer_properties.AggregateConformerPropertiesInterfaceConfig,
    ),
    "align": (
        align.main,
        align.AlignInterfaceConfig,
    ),
    "collect": (
        collect.main,
        collect.CollectInterfaceConfig,
    ),
}

# Commands that take top-level input_file and results_directory
_DIRECT_INPUT_COMMANDS = {"read_input", "add_reference_conformers"}

# Commands that take input.results_directory (BaseInterfaceConfig subclasses)
_INTERFACE_INPUT_COMMANDS = {
    "fast_conformers",
    "rescoss_conformers",
    "optimize_reference_conformers",
    "conformer_properties",
    "cosmotherm_properties",
    "generate_tautomers",
    "protonate",
    "cosmo_pka",
    "solvent_screening",
    "cocrystal_screening",
    "calculate_boltzmann_weights",
    "aggregate_conformer_properties",
    "align",
    "collect",
}


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------


class PipelineSubmissionConfig(SubmissionConfig):
    """UGE submission settings specific to the Pipeline CLI."""

    cores_per_task: int = Field(
        default=1,
        description="Number of cores to allocate per task",
        ge=1,
    )
    max_time: str = Field(
        default="3days",
        description="Maximum wall-clock time for the pipeline job (e.g., '10hours', '3days')",
    )
    name: str = Field(
        default="QMA_PIPELINE",
        description="Name for the pipeline job",
    )


class PipelineConfig(BaseModel):
    """Complete configuration for a pipeline run."""

    results_directory: Path | None = Field(
        default=None,
        description="Results directory (injected into all steps that need it)",
    )
    input_file: list[Path] | None = Field(
        default=None,
        description="Input molecule file(s) for read_input and add_reference_conformers steps",
    )
    reference_input_file: list[Path] | None = Field(
        default=None,
        description="Reference conformer input file(s) for add_reference_conformers "
        "(defaults to input_file if not provided)",
    )
    software_config: Path | None = Field(
        default=None,
        description="Path to software configuration YAML file (overrides qm_atlas_SOFTWARE_CONFIG_FILE env var)",
    )
    submission_config: SubmissionConfig = Field(
        default_factory=PipelineSubmissionConfig,
        description="UGE job submission settings for the pipeline itself",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )
    steps: list[Any] = Field(
        ...,
        description="Ordered list of pipeline steps to execute",
        min_length=1,
    )

    @field_validator("results_directory", mode="after")
    @classmethod
    def validate_results_directory_field(cls, v: Path | None) -> Path | None:
        """Validate results directory using arg_utils validation.

        For read_input, the results directory is created if it doesn't exist.
        Pydantic handles conversion to Path.
        """
        if v is None:
            return None
        validate_results_dir(v, create_if_missing=True)
        return v.expanduser().resolve()

    @field_validator("input_file", "reference_input_file", mode="after")
    @classmethod
    def validate_input_file(cls, v: list[Path] | None) -> list[Path] | None:
        """Resolve paths and validate existence and file type."""
        if v is None:
            return None
        paths = [p.expanduser().resolve() for p in v]

        for path in paths:
            if not path.exists():
                raise ValueError(f"Input path does not exist: {path}")
            if path.is_file() and path.suffix.lower() not in {".sdf", ".csv"}:
                raise ValueError(f"Unsupported input file type: {path} (must be .sdf or .csv)")

        return paths

    @staticmethod
    def _deserialize_step(v: Any) -> Any:
        """Convert dict to appropriate step config class based on command field.

        Similar to the ``_deserialize_config`` pattern used in
        ``ConformerPropertyOptions`` for ``calculation_tasks``.
        """
        if isinstance(v, dict):
            command = v.get("command", "undefined")
            if command not in STEP_REGISTRY:
                raise ValueError(
                    f"Unknown pipeline command: {command}. "
                    f"Available commands: {list(STEP_REGISTRY.keys())}"
                )
            _, config_cls = STEP_REGISTRY[command]
            return config_cls(**v)
        return v

    @staticmethod
    def _inject_paths_into_step(
        step_dict: dict[str, Any],
        *,
        results_directory: Path | None,
        input_file: list[Path] | None,
        reference_input_file: list[Path] | None,
        software_config: Path | None,
    ) -> None:
        """Inject top-level pipeline paths into a raw step dict.

        Values already present in the step dict take precedence (no overwrite).
        """
        command = step_dict.get("command")

        if software_config is not None and "software_config" not in step_dict:
            step_dict["software_config"] = software_config

        if command in _DIRECT_INPUT_COMMANDS:
            if results_directory is not None and "results_directory" not in step_dict:
                step_dict["results_directory"] = results_directory

            if command == "read_input":
                if input_file is not None and "input_file" not in step_dict:
                    step_dict["input_file"] = input_file
            elif command == "add_reference_conformers":
                if "input_file" not in step_dict:
                    if reference_input_file is not None:
                        step_dict["input_file"] = reference_input_file
                    elif input_file is not None:
                        step_dict["input_file"] = input_file

        elif command in _INTERFACE_INPUT_COMMANDS:
            if results_directory is not None:
                input_dict = step_dict.setdefault("input", {})
                if isinstance(input_dict, dict) and "results_directory" not in input_dict:
                    input_dict["results_directory"] = results_directory

    @staticmethod
    def _force_wait_on_step(step: Any) -> None:
        """Force wait=true on a step that has submit=true.

        Works on both raw dicts and typed config objects.
        """
        if isinstance(step, dict):
            sub_cfg = step.get("submission_config")
            if isinstance(sub_cfg, dict) and sub_cfg.get("submit", False):
                sub_cfg["wait"] = True
        elif hasattr(step, "submission_config"):
            if step.submission_config.submit:
                step.submission_config.wait = True

    @model_validator(mode="after")
    def resolve_steps(self) -> "PipelineConfig":
        """Inject paths, force wait, and deserialize steps to typed config objects.

        This runs after all field validators, so ``self.results_directory``
        etc. are already resolved.  For each raw step dict the method:

        1. Injects top-level pipeline paths (results_directory, input_file, …)
        2. Forces ``wait=true`` on steps with ``submit=true``
        3. Deserializes the dict into the appropriate config class via
           :attr:`STEP_REGISTRY`, following the same pattern used by
           ``ConformerPropertyOptions`` for ``calculation_tasks``.
        """
        resolved: list[Any] = []
        for step_data in self.steps:
            if isinstance(step_data, dict):
                self._inject_paths_into_step(
                    step_data,
                    results_directory=self.results_directory,
                    input_file=self.input_file,
                    reference_input_file=self.reference_input_file,
                    software_config=self.software_config,
                )
                self._force_wait_on_step(step_data)
                resolved.append(self._deserialize_step(step_data))
            else:
                # Already a typed config (e.g. from model_copy)
                self._force_wait_on_step(step_data)
                resolved.append(step_data)
        self.steps = resolved
        return self


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> PipelineConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides for paths and submission settings

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        PipelineConfig: Validated configuration object with paths resolved.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="Pipeline — Run a multi-step qm-atlas pipeline from a YAML config file",
        env_prefix="qm_atlas_PIPELINE_",
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to pipeline YAML config file",
    )

    parser.add_class_arguments(PipelineConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = PipelineConfig(**interface_namespace)
    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> None:
    """Execute all steps in the pipeline sequentially.

    Args:
        config: Validated pipeline configuration with paths and wait flags already resolved.
    """
    # Initialize pipeline log file
    log_file: Path | None = None
    if config.results_directory is not None:
        unique_name = uuid.uuid4().hex[:8]
        log_file = config.results_directory / f"pipeline_{unique_name}.log"

    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    level = logging.DEBUG if config.verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format=logging_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Snapshot the pipeline's own handlers so they can be reinstated after each
    # step. Locally-run steps redirect the root logger to per-task log files via
    # task_files.change_log_file(); without restoring, all subsequent
    # pipeline-level messages would leak into the last task's log file.
    pipeline_handlers = list(logging.root.handlers)
    pipeline_level = logging.root.level

    n_steps = len(config.steps)
    _logger.info(f"Starting pipeline with {n_steps} step(s)")
    if log_file is not None:
        _logger.info(f"Pipeline log file: {log_file}")

    for i, step in enumerate(config.steps, start=1):
        _logger.info(f"--- Pipeline step {i}/{n_steps}: {step.command} ---")

        main_func, _ = STEP_REGISTRY[step.command]

        # Call the command's main function with the typed step config directly
        main_func(config=step)

        # Reinstate the pipeline's logging in case the step ran locally and
        # redirected the root logger to per-task log files.
        task_files.reset_root_logging(pipeline_handlers, pipeline_level)

        _logger.info(f"--- Completed step {i}/{n_steps}: {step.command} ---")

    _logger.info("Pipeline finished successfully")


# ---------------------------------------------------------------------------
# Job Submission
# ---------------------------------------------------------------------------


def submit(config: PipelineConfig) -> None:
    """Submit the pipeline itself to the cluster.

    Dumps the fully resolved config to a YAML file, then submits
    ``qm-atlas pipeline --config <resolved_config>.yaml`` to the cluster.
    """
    if config.results_directory is None:
        raise ValueError("results_directory must be specified for pipeline submission")

    # Prepare scratch directory
    task_scratch_dir = get_submission_scratch_dir(config.results_directory)
    task_scratch_dir.mkdir(parents=True, exist_ok=True)

    task_uid = generate_name()

    # Create a copy with submit=False for the on-cluster execution
    submitted_config = config.model_copy(deep=True)
    submitted_config.submission_config = submitted_config.submission_config.model_copy(
        update={"submit": False}
    )

    # Dump resolved config to YAML (mode='json' ensures Path objects become strings)
    config_file = (task_scratch_dir / f"pipeline_config_{task_uid}.yaml").resolve()
    config_data = submitted_config.model_dump(mode="json")
    # Steps are already serialized with their command field included
    del config_data["submission_config"]  # Not needed in the submitted config

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    # Build the submission command
    submission_command = f"{COMMAND} --config {config_file}"
    hours, mins = parse_time(config.submission_config.max_time)
    log_dir = (config.results_directory / "log").expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Export software config env var in submission script if specified
    software_config_env_var_value = os.getenv(SOFTWARE_CONFIG_ENV_VAR, None)
    if software_config_env_var_value is not None:
        export_cmd = f"export {SOFTWARE_CONFIG_ENV_VAR}={software_config_env_var_value}"
        submission_command = f"{export_cmd}\n{submission_command}"

    environment_manager = SOFTWARE_CONFIG.get_environment_manager()
    env_submission_config = environment_manager.get_submission_config().copy()
    command_prepend = env_submission_config.pop("command_prepend", None)
    if command_prepend is not None:
        submission_command = f"{command_prepend} \n{submission_command}"

    user_email = config.submission_config.user_email

    submission_script = generate_script(
        cmd=submission_command,
        name=config.submission_config.name,
        cores=config.submission_config.cores_per_task,
        mem=config.submission_config.mem_per_core,
        hours=hours,
        mins=mins,
        log_dir=log_dir,
        user_email=user_email,
        **env_submission_config,
    )
    script_path = write_script(
        content=submission_script, directory=task_scratch_dir, filename=f"pipeline_{task_uid}.sh"
    )
    job_id = submit_script_with_retry(script_path)
    _logger.info(f"Pipeline job '{config.submission_config.name}' submitted with ID: {job_id}")

    if config.submission_config.wait:
        _logger.info("Waiting for pipeline job to complete...")
        wait_for_jobs_using_hold_job(
            [job_id],
            task_scratch_dir,
            user_email=config.submission_config.user_email,
            log_dir=log_dir,
            name=f"{config.submission_config.name}_hold",
        )
        _logger.info("Pipeline job finished.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: PipelineConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the pipeline CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (PipelineConfig | None):
            Optional PipelineConfig object. If provided, overrides command-line arguments.
        log_file (Path | None):
            Optional file to append logs to.
    """
    if config is None:
        config = load_config(argv=argv)
    apply_software_config(config.software_config)
    setup_logging(verbose=config.verbose, log_file=log_file)

    if config.submission_config.submit:
        submit(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
