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
from qm_atlas.workflows import rescoss_conformers

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas rescoss_conformers"

# ---------------------------------------------------------------------------
# ReSCoSS-specific Options
# ---------------------------------------------------------------------------


class RescossOptions(BaseModel):
    """Options specific to the ReSCoSS conformer generation workflow."""

    conformer_generation_options: ConformerGenerationOptions = Field(
        default=rescoss_conformers.RESCOSS_DEFAULT_CONF_GENS,
        description="Options for the initial conformer generation step, specifying which conformer generators to use and their settings.",
    )
    expand_conformers: bool = Field(
        default=True,
        description="Whether to perform force-field based conformer expansion",
    )
    xtb_optimization_config: list[OptimizationConfig] = Field(
        default=rescoss_conformers.XTB_OPTIMIZATION_CONFIG,
        description="Settings for the intermediate geometry optimization at the xTB level. Performed on all conformers, before clustering and selection.",
    )
    rmsd_threshold: float = Field(
        default=rescoss_conformers.RMSD_THRESHOLD,
        description="RMSD threshold for conformer deduplication after the xtb geometry optimization and before clustering",
        gt=0,
    )
    num_clusters: int = Field(
        default=rescoss_conformers.DEFAULT_NUM_CLUSTERS,
        description="Number of clusters for k-means clustering",
        ge=1,
    )
    num_per_cluster: int = Field(
        default=rescoss_conformers.DEFAULT_NUM_PER_CLUSTER,
        description="Number of conformers to keep per cluster and solvent",
        ge=1,
    )
    final_optimization_config: list[OptimizationConfig] = Field(
        default=rescoss_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG,
        description="Settings for the final geometry optimization at the DFT level. Performed on selected conformers after clustering.",
    )
    final_rmsd_threshold: float = Field(
        default=rescoss_conformers.RMSD_THRESHOLD / 5,
        description="RMSD threshold for final conformer deduplication after DFT optimization",
        gt=0,
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

    @field_validator("xtb_optimization_config", "final_optimization_config", mode="before")
    @classmethod
    def deserialize_configs(cls, v: Any) -> Any:
        """Deserialize optimization config sequences from dicts."""
        if isinstance(v, list):
            return [cls._deserialize_config(item) for item in v]
        return v


# ---------------------------------------------------------------------------
# Different Defaults for Submitted Jobs
# ---------------------------------------------------------------------------


class RescossSubmissionConfig(SubmissionConfig):
    """UGE submission settings specific to the ReSCoSS CLI."""

    # Inherit all fields from SubmissionConfig, override certain default values for this workflow
    cores_per_task: int = Field(
        default=8,
        description="Number of cores to allocate per task",
        ge=1,
    )
    max_time: str = Field(
        default="3days",
        description="Maximum wall-clock time per job (e.g., '10hours', '2days')",
    )
    name: str = Field(
        default="QMA_RESCOSS",
        description="Name for the job",
    )


# ---------------------------------------------------------------------------
# Top-level Configuration
# ---------------------------------------------------------------------------


class RescossInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for ReSCoSS conformer generation."""

    command: Literal["rescoss_conformers"] = Field(
        default="rescoss_conformers",
        description="CLI command name",
    )
    submission_config: RescossSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=RescossSubmissionConfig,
        description="UGE job submission settings",
    )
    rescoss_options: RescossOptions = Field(
        default_factory=RescossOptions,
        description="ReSCoSS workflow specific options",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> RescossInterfaceConfig:
    """Load and validate configuration using jsonargparse + Pydantic.

    Supports:
    - YAML config file via --config
    - Command-line argument overrides
    - Automatic type coercion

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        RescossInterfaceConfig: Validated configuration object.

    Raises:
        SystemExit: On parse errors or validation failures.
    """
    parser = ArgumentParser(
        description="ReSCoSS Conformer Generation — Configure via YAML or CLI arguments.\n"
        "Run 'qm-atlas describe optimizers' and 'qm-atlas describe conformer_generators' "
        "to see available options.",
        env_prefix="qm_atlas_RESCOSS_",
    )

    # Add --config to load from YAML file
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to YAML config file",
    )

    # Add the full RescossInterfaceConfig as structured arguments
    parser.add_class_arguments(RescossInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    config = RescossInterfaceConfig(**interface_namespace)

    return config


# ---------------------------------------------------------------------------
# Core Workflow Functions
# ---------------------------------------------------------------------------


def run_local(config: RescossInterfaceConfig) -> None:
    """Run ReSCoSS conformer generation locally.

    Args:
        config: RescossInterfaceConfig with all settings.
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
                _logger.error(f"Conformer Generation failed for {sdf_file}")
                _logger.error(f"Got Exception {exc}")
                _logger.error(traceback.format_exc())

            finally:
                stop_time = time.perf_counter()
                eval_time = stop_time - start_time
                _logger.info(f"Evaluated conformer generation in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def run_job(cpd_dir: compound_dir.CpdDir, sdf_file: Path, config: RescossInterfaceConfig) -> None:

    _logger.info(f"Working on Molecule {sdf_file.stem}")
    mols = input_file.read_input_sdf(sdf_file)

    if len(mols) > 1:
        _logger.warning("Received more than one input molecule. Proceeding with first")

    mol = mols[0]
    rescoss_options = config.rescoss_options
    mol_3d = rescoss_conformers.generate_rescoss_conformers(
        mol,
        conformer_generation_options=rescoss_options.conformer_generation_options,
        expand_conformers=rescoss_options.expand_conformers,
        xtb_optimization_config=rescoss_options.xtb_optimization_config,
        rmsd_threshold=rescoss_options.rmsd_threshold,
        num_clusters=rescoss_options.num_clusters,
        num_per_cluster=rescoss_options.num_per_cluster,
        final_optimization_config=rescoss_options.final_optimization_config,
        final_rmsd_threshold=rescoss_options.final_rmsd_threshold,
        write_intermediates=rescoss_options.write_intermediates,
        n_cores=config.n_cores or 1,
    )

    mol_3d.SetProp("_Name", cpd_dir.find_state_by_file(sdf_file))
    cpd_dir.add_result_conformers(mol_3d)


# ---------------------------------------------------------------------------
# Job Submission
# ---------------------------------------------------------------------------


def submit(config: RescossInterfaceConfig) -> None:
    """Submit ReSCoSS conformer generation jobs to cluster."""
    # Extract worker paths (molecules to process)
    # Skip compounds whose conformers already exist (resume a partial run)
    worker_paths = extract_worker_paths(
        config.input, workflow_type="conformer_expansion", skip_completed=True
    )

    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} molecules to process")
    _logger.info("Submitting Task Array for ReSCoSS Conformer Generation")
    _logger.info(f"UGE Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

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
    config: RescossInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
    """Main entry point for the ReSCoSS CLI.

    Args:
        argv (list[str] | None):
            List of command-line arguments (for testing). If None, uses sys.argv[1:].
        config (RescossInterfaceConfig | None):
            Optional RescossInterfaceConfig object. If provided, overrides command-line arguments.
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
