"""ReSCoSS Conformer Generation CLI.

Perform the ReSCoSS conformer generation workflow on a collection of molecules.
"""
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, field_validator

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
    worker_path_completed,
)
from qm_atlas.command_line.file_interface import compound_dir, input_file, task_files
from qm_atlas.command_line.utils import submission
from qm_atlas.command_line.utils.check_results import check_conformer_expansion
from qm_atlas.tasks.conformer_generation import ConformerGenerationOptions
from qm_atlas.tasks.optimize import (
    JobexOptions,
    OptimizationConfig,
    XtbOptions,
    XtbTurbomoleOptions,
)
from qm_atlas.workflows import fast_conformers

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas fast_conformers"

# ---------------------------------------------------------------------------
# Fast Conformers-specific Options
# ---------------------------------------------------------------------------


class FastConformersOptions(BaseModel):
    """Options specific to the Fast Conformers workflow."""

    conformer_generation_options: ConformerGenerationOptions = Field(
        default=fast_conformers.FAST_DEFAULT_OPTIONS,
        description="Options for the initial conformer generation step, specifying which conformer generators to use and their settings.",
    )
    expand_conformers: bool = Field(
        default=True,
        description="Whether to perform force-field based conformer expansion",
    )
    optimization_config: list[OptimizationConfig] = Field(
        default=fast_conformers.DEFAULT_OPTIMIZATION_CONFIG,
        description="Settings for the intermediate geometry optimization at the xTB level",
    )
    final_optimization_config: list[OptimizationConfig] = Field(
        default=fast_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG,
        description="Settings for the final geometry optimization at the DFT level. This can be used to run a more expensive optimization on the final conformers. Skipped by default.",
    )
    write_intermediates: bool = Field(
        default=False,
        description="Write intermediate SDF files at each workflow step (debugging)",
    )

    @staticmethod
    def _deserialize_config(v: Any) -> Any:
        """Convert dict to appropriate options class based on the backend field."""
        if isinstance(v, dict):
            backend = v.get("backend", "xtb_turbomole")
            if backend == "xtb":
                return XtbOptions(**v)
            elif backend == "jobex":
                return JobexOptions(**v)
            elif backend == "xtb_turbomole":
                return XtbTurbomoleOptions(**v)
            else:
                raise ValueError(f"Unknown optimization backend: {backend}")
        return v

    @field_validator("optimization_config", "final_optimization_config", mode="before")
    @classmethod
    def deserialize_configs(cls, v: Any) -> Any:
        """Deserialize optimization config sequences from dicts."""
        if isinstance(v, list):
            return [cls._deserialize_config(item) for item in v]
        return v


# ---------------------------------------------------------------------------
# Different Defaults for Submitted Jobs
# ---------------------------------------------------------------------------


class FastConformersSubmissionConfig(SubmissionConfig):
    """UGE submission settings specific to the Fast Conformers CLI."""

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
        default="QMA_FAST_CONF",
        description="Name for the job",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class FastConformersInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for Fast Conformers workflow."""

    command: Literal["fast_conformers"] = Field(
        default="fast_conformers",
        description="CLI command name",
    )
    submission_config: FastConformersSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=FastConformersSubmissionConfig,
        description="UGE job submission settings",
    )
    fast_conformers_options: FastConformersOptions = Field(
        default_factory=FastConformersOptions,
        description="Fast Conformers workflow specific options",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> FastConformersInterfaceConfig:
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
        description="Fast Conformers Workflow — Configure via YAML or CLI arguments.\n"
        "Run 'qm-atlas describe optimizers' and 'qm-atlas describe conformer_generators' "
        "to see available options.",
        env_prefix="qm_atlas_FAST_CONFORMERS_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full FastConformersInterfaceConfig as structured arguments
    parser.add_class_arguments(FastConformersInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = FastConformersInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_local(config: FastConformersInterfaceConfig) -> None:
    """Run Fast Conformers workflow locally.

    Args:
        config: FastConformersInterfaceConfig with all settings.
    """

    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="conformer_expansion")

    prev_log_file = None
    try:
        for cpd_dir, sdf_file, log_file in worker_paths:

            if worker_path_completed(cpd_dir, sdf_file, "conformer_expansion"):
                _logger.info(f"Skipping {sdf_file.stem}: results already present")
                continue

            # use new log file, if required
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file

            start_time = time.perf_counter()

            try:
                run_job(cpd_dir, sdf_file, config)

            except KeyboardInterrupt:
                _logger.error("Got ^C while running conformer generation jobs.")
                sys.exit()

            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Fast Conformers workflow failed for {sdf_file}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())

            finally:
                stop_time = time.perf_counter()
                eval_time = stop_time - start_time
                _logger.info(f"Evaluated Fast Conformers workflow in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def run_job(
    cpd_dir: compound_dir.CpdDir, sdf_file: Path, config: FastConformersInterfaceConfig
) -> None:

    _logger.info(f"Working on Molecule {sdf_file.stem}")
    mols = input_file.read_input_sdf(sdf_file)

    if len(mols) > 1:
        _logger.warning("Received more than input molecule. Proceeding with first")

    mol = mols[0]
    fast_conformers_options = config.fast_conformers_options

    mol_3d = fast_conformers.generate_fast_conformers(
        mol,
        conformer_generation_options=fast_conformers_options.conformer_generation_options,
        expand_conformers=fast_conformers_options.expand_conformers,
        optimization_config=fast_conformers_options.optimization_config,
        final_optimization_config=fast_conformers_options.final_optimization_config,
        write_intermediates=fast_conformers_options.write_intermediates,
        n_cores=config.n_cores or 1,
    )

    mol_3d.SetProp("_Name", cpd_dir.find_state_by_file(sdf_file))
    cpd_dir.add_result_conformers(mol_3d)


# ---------------------------------------------------------------------------
# Job Submission
# ---------------------------------------------------------------------------


def submit(config: FastConformersInterfaceConfig) -> None:
    """Submit Fast Conformers jobs to cluster."""
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

    # Extract worker paths, skipping compounds whose conformers already exist (resume)
    worker_paths = extract_worker_paths(
        config.input, workflow_type="conformer_expansion", skip_completed=True
    )

    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} molecules to process")

    submission.submit(
        worker_paths=worker_paths,
        config=config,
        command=COMMAND,
    )

    if config.submission_config.wait:
        check_conformer_expansion(worker_paths)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: FastConformersInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the Fast Conformers CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (FastConformersInterfaceConfig | None):
            Optional FastConformersInterfaceConfig object. If provided, overrides command-line arguments.
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
        run_local(config)


if __name__ == "__main__":
    main()
