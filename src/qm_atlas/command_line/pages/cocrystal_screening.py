"""COSMOtherm cocrystal screening CLI.

Runs COSMOtherm cocrystal screening on previously generated ``.cosmo`` files,
wrapping :func:`cosmo_cocrystal.screen_cocrystals`. Results from every run are
appended to a single ``{cpd_name}_cocrystal_screening.csv`` file inside each
compound directory.
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, model_validator

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.utils import submission
from qm_atlas.wrappers.cosmotherm import cosmo_cocrystal

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas cocrystal_screening"

STATES_COL = "states"


class CocrystalScreeningOptions(BaseModel):
    """Options for the cocrystal screening workflow."""

    coformers: list[str] = Field(
        default_factory=list,
        description="Names of coformers to screen. Corresponding .cosmo files must be findable.",
    )
    level: str = Field(default="bp-tzvpd", description="COSMO level of theory.")
    coformers_dirs: list[Path] | None = Field(
        default=None,
        description="Additional directories searched for coformer .cosmo files.",
    )
    temperature_Celsius: float = Field(
        default=25.0, description="Temperature for the calculation in degrees Celsius."
    )
    stoichiometry: tuple[int, int] = Field(
        default=(1, 1),
        description="Cocrystal stoichiometry (API, coformer).",
    )
    f_fit: bool | tuple[float, float, float] = Field(
        default=False,
        description=(
            "Add 'f_fit' to the COSMOtherm action line. Pass True for default parameters "
            "or a 3-tuple to set them."
        ),
    )
    f_fit_cof: bool | tuple[float, float] = Field(
        default=True,
        description=(
            "Add 'f_fit_cof' to the COSMOtherm action line. Pass True for default "
            "parameters or a 2-tuple to set them."
        ),
    )
    f_fit_sol: bool | tuple[float, float, float] = Field(
        default=False,
        description=(
            "Add 'f_fit_sol' to the COSMOtherm action line. Pass True for default "
            "parameters or a 3-tuple to set them."
        ),
    )
    pZWI: bool = Field(
        default=False,
        description="Add 'pZWI' to the COSMOtherm action line (punish zwitterions).",
    )
    use_config_cosmo_dir: bool = Field(
        default=True,
        description="Whether to use the COSMO directory configured via the software config.",
    )

    @model_validator(mode="after")
    def validate_stoichiometry(self) -> "CocrystalScreeningOptions":
        if any(s <= 0 for s in self.stoichiometry):
            raise ValueError(
                f"Stoichiometry entries must be positive, got {self.stoichiometry!r}."
            )
        return self


class CocrystalScreeningSubmissionConfig(SubmissionConfig):
    cores_per_task: int = Field(default=1, ge=1)
    max_time: str = Field(default="4hours")
    name: str = Field(default="QMA_CocrystalScreen")


class CocrystalScreeningInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for the cocrystal screening workflow."""

    command: Literal["cocrystal_screening"] = Field(default="cocrystal_screening")
    submission_config: CocrystalScreeningSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=CocrystalScreeningSubmissionConfig,
    )
    cocrystal_screening_options: CocrystalScreeningOptions = Field(
        default_factory=CocrystalScreeningOptions,
        description="Cocrystal screening workflow options.",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> CocrystalScreeningInterfaceConfig:
    parser = ArgumentParser(
        description=(
            "COSMOtherm cocrystal screening workflow on previously generated .cosmo files."
        ),
        env_prefix="qm_atlas_COCRYSTAL_SCREENING_",
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to YAML config file")
    parser.add_class_arguments(CocrystalScreeningInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    return CocrystalScreeningInterfaceConfig(**interface_namespace)


# ---------------------------------------------------------------------------
# CSV append helper
# ---------------------------------------------------------------------------


def _append_results(
    cpd_dir: compound_dir.CpdDir,
    df: pd.DataFrame,
    *,
    state_names: list[str],
) -> None:
    if df is None or df.empty:
        _logger.warning(f"Empty cocrystal-screening result for states {state_names}.")
        return

    annotated = df.copy()
    annotated.insert(0, STATES_COL, ",".join(state_names))

    cpd_dir.append_cocrystal_results(annotated)


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_local(config: CocrystalScreeningInterfaceConfig) -> None:
    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options: dict[str, Any] = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="cocrystal_screening")

    prev_log_file = None
    try:
        for cpd_dir, _sdf_file, log_file in worker_paths:
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file

            start_time = time.perf_counter()
            try:
                run_job(cpd_dir, config)
            except KeyboardInterrupt:
                _logger.error("Got ^C while running cocrystal screening jobs.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Cocrystal screening failed for {cpd_dir.cpd_name}: {exc}")
                _logger.error(traceback.format_exc())
            finally:
                eval_time = time.perf_counter() - start_time
                _logger.info(f"Evaluated cocrystal screening in {eval_time:.2f} seconds")
    finally:
        task_files.close_task_log_handler()


def _collect_neutral_cosmo_files(
    cpd_dir: compound_dir.CpdDir,
) -> tuple[list[str], list[Path]]:
    """Return the neutral (charge=0) state names and all of their COSMO files."""
    neutral_states = cpd_dir.registry_handler.get_registered_names(charge=0)
    cosmo_files: list[Path] = []
    for state_name in neutral_states:
        for result_sdf in cpd_dir.get_result_files(state_name):
            cosmo_file = cpd_dir.get_associated_cosmo_file(result_sdf)
            if not cosmo_file.exists():
                _logger.warning(f"Could not find COSMO file for {result_sdf}, skipping")
                continue
            cosmo_files.append(cosmo_file)
    return neutral_states, cosmo_files


def run_job(
    cpd_dir: compound_dir.CpdDir,
    config: CocrystalScreeningInterfaceConfig,
) -> None:
    _logger.info(f"Working on compound {cpd_dir.cpd_name}")
    options = config.cocrystal_screening_options

    if not options.coformers:
        _logger.error("No coformers configured for screening; set 'coformers'.")
        return

    neutral_states, cosmo_files = _collect_neutral_cosmo_files(cpd_dir)

    if not neutral_states:
        _logger.error(
            f"No neutral (charge=0) states registered for {cpd_dir.cpd_name}; "
            "skipping cocrystal screening."
        )
        return

    if not cosmo_files:
        _logger.error(
            f"No COSMO files available for neutral states of {cpd_dir.cpd_name}; "
            "skipping cocrystal screening."
        )
        return

    _logger.info(
        f"Combining {len(cosmo_files)} neutral-state COSMO conformers "
        f"({len(neutral_states)} states: {neutral_states}) into one screening call."
    )
    _logger.info(
        f"Running cocrystal screening for {cpd_dir.cpd_name} over {len(options.coformers)} coformers."
    )
    try:
        df = cosmo_cocrystal.screen_cocrystals(
            compound_conformers=list(cosmo_files),
            coformers=list(options.coformers),
            level=options.level,
            coformers_dirs=options.coformers_dirs,
            temperature_Celsius=options.temperature_Celsius,
            stoichiometry=options.stoichiometry,
            f_fit=options.f_fit,
            f_fit_cof=options.f_fit_cof,
            f_fit_sol=options.f_fit_sol,
            pZWI=options.pZWI,
            n_cores=config.n_cores or 1,
            use_config_cosmo_dir=options.use_config_cosmo_dir,
        )
    except Exception as exc:  # pylint: disable=broad-except
        _logger.error(f"Cocrystal screening failed for {cpd_dir.cpd_name}: {exc}")
        _logger.error(traceback.format_exc())
        return

    _append_results(cpd_dir, df, state_names=neutral_states)


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def submit(config: CocrystalScreeningInterfaceConfig) -> None:
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    worker_paths = extract_worker_paths(config.input, workflow_type="cocrystal_screening")
    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} molecules to process")
    submission.submit(worker_paths=worker_paths, config=config, command=COMMAND)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: CocrystalScreeningInterfaceConfig | None = None,
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
