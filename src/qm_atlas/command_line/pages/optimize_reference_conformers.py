import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import Field, field_validator

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
    worker_path_completed,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.utils import submission
from qm_atlas.command_line.utils.check_results import check_reference_optimization
from qm_atlas.tasks.optimize import (
    JobexOptions,
    OptimizationConfig,
    XtbOptions,
    XtbTurbomoleOptions,
)
from qm_atlas.workflows import optimize_constrained

_logger = logging.getLogger(__name__)


COMMAND = "qm-atlas optimize_reference_conformers"


# ---------------------------------------------------------------------------
# Different Defaults for Submitted Jobs
# ---------------------------------------------------------------------------


class ReferenceOptimizationSubmissionConfig(SubmissionConfig):
    """UGE submission settings specific to the ReSCoSS CLI."""

    # Inherit all fields from SubmissionConfig, override certain default values for this workflow
    cores_per_task: int = Field(
        default=2,
        description="Number of cores to allocate per task (one task are all the constrained optimizations of one reference conformer). Recommended to use as many cores as different constrained optimizations settings. Defaults to 2, correspondign to the two force constants used in the default optimization settings.",
        ge=1,
    )
    max_time: str = Field(
        default="1day",
        description="Maximum wall-clock time per job (e.g., '10hours', '2days')",
    )
    name: str = Field(
        default="QMA_REF_OPT",
        description="Name for the job",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class RefInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for reference conformer generation."""

    command: Literal["optimize_reference_conformers"] = Field(
        default="optimize_reference_conformers",
        description="CLI command name",
    )
    submission_config: ReferenceOptimizationSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=ReferenceOptimizationSubmissionConfig,
        description="UGE job submission settings",
    )
    constrained_optimization_config: list[list[OptimizationConfig]] = Field(
        default=optimize_constrained.DEFAULT_OPTIMIZATION_CONFIG,
        description="Options for the constrained optimizations of the reference conformations",
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

    @field_validator("constrained_optimization_config", mode="before")
    @classmethod
    def deserialize_configs(cls, v: Any) -> Any:
        """Deserialize optimization config sequences from dicts."""
        if isinstance(v, list):
            return [[cls._deserialize_config(item) for item in sublist] for sublist in v]
        return v


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> RefInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        RefInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="ReSCoSS Conformer Generation — Configure via YAML or CLI arguments.\n"
        "Run 'qm-atlas describe optimizers' to see available optimization drivers and their options.",
        env_prefix="qm_atlas_RESCOSS_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full RefInterfaceConfig as structured arguments
    parser.add_class_arguments(RefInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = RefInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_local(config: RefInterfaceConfig) -> None:
    """Run constrained optimization locally.

    Args:
        config: RefInterfaceConfig with all settings.
    """

    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="reference_optimization")

    prev_log_file = None
    try:
        for cpd_dir, sdf_file, log_file in worker_paths:

            if worker_path_completed(cpd_dir, sdf_file, "reference_optimization"):
                _logger.info(f"Skipping {sdf_file.stem}: reference results already present")
                continue

            # use new log file, if required
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file

            start_time = time.perf_counter()

            try:
                run_job(cpd_dir, sdf_file, config)

            except KeyboardInterrupt:
                _logger.error("Got ^C while running constrained optimization jobs.")
                sys.exit()

            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Constrained Optimization failed for {sdf_file}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())

            finally:
                stop_time = time.perf_counter()
                eval_time = stop_time - start_time
                _logger.info(f"Evaluated constrained optimization in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def run_job(cpd_dir: compound_dir.CpdDir, sdf_file: Path, config: RefInterfaceConfig) -> None:

    _logger.info(f"Working on Molecule {sdf_file.stem}")
    mol = cpd_dir.extract_mol(sdf_file)

    optimized_mols = optimize_constrained.run_constrained_optimization(
        mol,
        n_cores=config.n_cores or 1,
        optimization_config=config.constrained_optimization_config,
    )

    num_saved = 0
    for optimized_mol in optimized_mols:
        # Failed optimizations return a molecule with no conformers (the failed
        # conformer is dropped by update_mol). Skip these instead of passing them
        # to add_reference_result, which requires exactly one conformer and would
        # otherwise raise an error that masks the underlying optimization failure.
        if optimized_mol.GetNumConformers() != 1:
            _logger.warning(
                f"Skipping a constrained optimization result for {sdf_file.stem} "
                "because it did not produce exactly one conformer (optimization failed)."
            )
            continue
        cpd_dir.add_reference_result(optimized_mol, sdf_file)
        num_saved += 1

    if num_saved == 0:
        raise RuntimeError(
            f"All constrained optimizations failed for {sdf_file}; "
            "no reference result was saved."
        )


def submit(config: RefInterfaceConfig) -> None:
    """Submit reference conformer optimization jobs to cluster."""
    _logger.info("Submitting Task Array for Reference Conformer Optimization")
    _logger.info(f"UGE Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

    # Skip reference conformers already optimized (resume a partial run)
    worker_paths = extract_worker_paths(
        config.input, workflow_type="reference_optimization", skip_completed=True
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
        check_reference_optimization(worker_paths)


def main(
    argv: list[str] | None = None,
    config: RefInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the Reference Conformer Optimization CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (RefInterfaceConfig | None):
            Optional RefInterfaceConfig object. If provided, overrides command-line arguments.
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
