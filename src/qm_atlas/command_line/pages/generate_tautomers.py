"""Generate Tautomers CLI.

For each existing input state, generate tautomers via the score_tautomers
workflow and register the surviving ones as additional input states in the
same compound directory.
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, input_file, task_files
from qm_atlas.command_line.utils import submission
from qm_atlas.tasks import conformer_generation
from qm_atlas.tasks.calculate_properties import TurbomoleSinglePointOptions
from qm_atlas.tasks.optimize import (
    JobexOptions,
    OptimizationConfig,
    XtbOptions,
    XtbTurbomoleOptions,
)
from qm_atlas.workflows import fast_conformers, score_tautomers
from qm_atlas.wrappers.cosmotherm import cosmo_tasks
from qm_atlas.wrappers.cosmotherm.cosmo_utils import CosmoSettings

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas generate_tautomers"


class GenerateTautomersOptions(BaseModel):
    """Options for the generate_tautomers workflow."""

    energy_threshold: float = Field(
        default=7.0,
        description="Energy cutoff (kcal/mol) vs. the lowest-energy tautomer; "
        "tautomers above this threshold are discarded.",
    )
    solvent_name: str = Field(
        default="h2o",
        description="COSMOtherm solvent used for tautomer scoring.",
    )
    single_point_config: TurbomoleSinglePointOptions = Field(
        default=score_tautomers.TAUTOMER_TM_SP_CONFIG,
        description="Turbomole single-point settings used to compute tautomer energies.",
    )
    cosmo_options: CosmoSettings = Field(
        default=cosmo_tasks.DESCRIPTORS_SETTINGS_SIMPLIFIED,
        description="COSMOtherm settings used for tautomer scoring.",
    )
    solvents_dirs: list[Path] | None = Field(
        default=None,
        description="Optional list of directories with COSMO files for the solvent.",
    )
    use_config_cosmo_dir: bool = Field(
        default=True,
        description="Whether to use the solvent COSMO directory from the software config.",
    )
    conformer_generation_options: conformer_generation.ConformerGenerationOptions = Field(
        default=fast_conformers.FAST_DEFAULT_OPTIONS,
        description="Force-field conformer generator settings used for tautomer conformer expansion.",
    )
    optimization_config: list[OptimizationConfig] = Field(
        default=fast_conformers.DEFAULT_OPTIMIZATION_CONFIG,
        description="Sequence of optimization steps applied after the first shape filter.",
    )
    final_optimization_config: list[OptimizationConfig] = Field(
        default=fast_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG,
        description="Optional sequence of optimization steps applied to the final unique conformers.",
    )

    model_config = {"arbitrary_types_allowed": True}

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


class GenerateTautomersSubmissionConfig(SubmissionConfig):
    """UGE submission settings specific to the generate_tautomers CLI."""

    cores_per_task: int = Field(default=4, ge=1)
    max_time: str = Field(default="1day")
    name: str = Field(default="QMA_TAUTOMERS")


class GenerateTautomersInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for the generate_tautomers workflow."""

    command: Literal["generate_tautomers"] = Field(default="generate_tautomers")
    submission_config: GenerateTautomersSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=GenerateTautomersSubmissionConfig,
    )
    generate_tautomers_options: GenerateTautomersOptions = Field(
        default_factory=GenerateTautomersOptions,
    )


def load_config(argv: list[str] | None = None) -> GenerateTautomersInterfaceConfig:
    parser = ArgumentParser(
        description="Generate tautomers and register them as new states in compound directories.",
        env_prefix="qm_atlas_GENERATE_TAUTOMERS_",
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to YAML config file")
    parser.add_class_arguments(GenerateTautomersInterfaceConfig)
    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    return GenerateTautomersInterfaceConfig(**interface_namespace)


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)


def _next_free_tautomer_name(parent_state_name: str, registered: set[str]) -> str:
    """Return the next free `{parent}_T{n}` name (n starts at 1)."""
    n = 1
    while True:
        candidate = f"{parent_state_name}_T{n}"
        if candidate not in registered:
            return candidate
        n += 1


def run_job(
    cpd_dir: compound_dir.CpdDir,
    sdf_file: Path,
    config: GenerateTautomersInterfaceConfig,
) -> None:
    _logger.info(f"Generating tautomers for {sdf_file.stem}")

    mols = input_file.read_input_sdf(sdf_file)
    if len(mols) == 0:
        _logger.warning(f"No molecules read from {sdf_file}; skipping.")
        return
    if len(mols) > 1:
        _logger.warning(f"{sdf_file} contains more than one molecule; using the first only.")
    parent_mol = mols[0]
    parent_state_name = cpd_dir.find_state_by_file(sdf_file)

    options = config.generate_tautomers_options
    tautomers = score_tautomers.generate_tautomers(
        parent_mol,
        energy_threshold=options.energy_threshold,
        solvent_name=options.solvent_name,
        single_point_config=options.single_point_config,
        cosmo_options=options.cosmo_options,
        solvents_dirs=options.solvents_dirs,
        use_config_cosmo_dir=options.use_config_cosmo_dir,
        conformer_generation_options=options.conformer_generation_options,
        optimization_config=options.optimization_config,
        final_optimization_config=options.final_optimization_config,
        n_cores=config.n_cores or 1,
        scr=cpd_dir.get_trace_dir(),
    )

    existing_smiles = set(cpd_dir.get_registered_smiles())
    registered_names = set(cpd_dir.registry_handler.get_registered_names())

    n_duplicates = 0
    n_new = 0
    for tautomer in tautomers:
        smi = _canonical_smiles(tautomer)
        if smi in existing_smiles:
            n_duplicates += 1
            continue

        new_name = _next_free_tautomer_name(parent_state_name, registered_names)
        tautomer.SetProp("_Name", new_name)
        try:
            cpd_dir.add_input_structure(
                tautomer,
                overwrite=False,
                description=f"tautomer generated from {parent_state_name}",
            )
        except (OSError, ValueError) as exc:
            _logger.error(f"Failed to register tautomer {new_name}: {exc}")
            continue

        existing_smiles.add(smi)
        registered_names.add(new_name)
        n_new += 1

    _logger.info(
        f"Tautomer scoring for {parent_state_name}: {len(tautomers)} survived scoring, "
        f"{n_new} registered as new states, {n_duplicates} skipped as duplicates."
    )


def run_local(config: GenerateTautomersInterfaceConfig) -> None:
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
        for cpd_dir_obj, sdf_file, log_file in worker_paths:
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file

            start_time = time.perf_counter()
            try:
                run_job(cpd_dir_obj, sdf_file, config)
            except KeyboardInterrupt:
                _logger.error("Got ^C while running tautomer generation jobs.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"generate_tautomers failed for {sdf_file}: {exc}")
                _logger.error(traceback.format_exc())
            finally:
                _logger.info(
                    f"Evaluated generate_tautomers in {time.perf_counter() - start_time:.2f} s"
                )
    finally:
        task_files.close_task_log_handler()


def submit(config: GenerateTautomersInterfaceConfig) -> None:
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    _logger.info(f"Cores per task: {config.submission_config.cores_per_task}")
    _logger.info(f"Max time: {config.submission_config.max_time}")

    worker_paths = extract_worker_paths(config.input, workflow_type="conformer_expansion")
    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} input states to process")

    submission.submit(
        worker_paths=worker_paths,
        config=config,
        command=COMMAND,
    )


def main(
    argv: list[str] | None = None,
    config: GenerateTautomersInterfaceConfig | None = None,
    log_file: Path | None = None,
) -> None:
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
