"""COSMOtherm solvent screening CLI.

Runs pure-solvent and/or binary/ternary mixture solubility screening using
previously generated ``.cosmo`` files. Results from every run are appended to
a single ``{cpd_name}_solubility_screening.csv`` file inside each compound
directory.
"""

import json
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
from qm_atlas.wrappers.cosmotherm import cosmo_solubility

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas solvent_screening"

# Bookkeeping columns prepended to every appended row.
STATES_COL = "states"
CALC_TYPE_COL = "calculation_type"
MODE_COL = "mode"
REFERENCES_COL = "references_json"


class ReferenceSolubilityConfig(BaseModel):
    """Pydantic mirror of :class:`cosmo_solubility.ReferenceSolubility`."""

    value: float = Field(description="Reference solubility value.")
    solvent_names: list[str] = Field(
        description="Names of the solvents the reference value refers to.",
    )
    temperature: float = Field(description="Reference temperature in degrees Celsius.")
    mass_fractions: list[float] | None = Field(
        default=None,
        description="Mass fractions of the reference solvents (only required for mixtures).",
    )
    solubility_unit: str = Field(
        default="c",
        description="Unit of the reference solubility (cosmotherm key).",
    )

    def to_dataclass(self) -> cosmo_solubility.ReferenceSolubility:
        return cosmo_solubility.ReferenceSolubility(
            value=self.value,
            solvent_names=list(self.solvent_names),
            temperature=self.temperature,
            mass_fractions=(
                list(self.mass_fractions) if self.mass_fractions is not None else None
            ),
            solubility_unit=self.solubility_unit,
        )


class SolventScreeningOptions(BaseModel):
    """Options for the solvent screening workflow."""

    solvent_names: list[str] = Field(
        default_factory=list,
        description="Pure solvents to screen via calculate_solubility.",
    )
    solvent_combinations: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Binary or ternary solvent combinations (lists of 2 or 3 solvent names) "
            "to screen via calculate_solubility_mixture."
        ),
    )
    melting_temperature_C: float | None = Field(
        default=None,
        description="Experimental melting temperature in degrees Celsius.",
    )
    melting_enthalpy_kJ_per_mol: float | None = Field(
        default=None,
        description="Experimental melting enthalpy in kJ/mol.",
    )
    level: str = Field(default="bp-tzvpd", description="COSMO level of theory.")
    references: list[ReferenceSolubilityConfig] = Field(
        default_factory=list,
        description="Reference solubilities used to anchor 'absolute' calculations.",
    )
    temperature: float = Field(
        default=25.0,
        description="Temperature for the solubility calculation in degrees Celsius.",
    )
    output_units: str = Field(
        default="wsolout_c lsolout_gl",
        description="COSMOtherm output unit string.",
    )
    num_steps_binary: int = Field(
        default=cosmo_solubility.NUM_STEPS_BINARY_MIXTURE,
        description="Number of mixing-ratio steps for binary mixtures.",
        ge=1,
    )
    num_steps_ternary: int = Field(
        default=cosmo_solubility.NUM_STEPS_TERNARY_MIXTURE,
        description="Number of mixing-ratio steps along each axis for ternary mixtures.",
        ge=1,
    )
    optimize: bool = Field(
        default=False,
        description="Ask COSMOtherm to optimise mixture ratios instead of screening them.",
    )
    solvents_dirs: list[Path] | None = Field(
        default=None,
        description="Additional directories searched for solvent .cosmo files.",
    )
    use_config_cosmo_dir: bool = Field(
        default=True,
        description="Whether to use the COSMO directory configured via the software config.",
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> "SolventScreeningOptions":
        for combo in self.solvent_combinations:
            if len(combo) not in (2, 3):
                raise ValueError(
                    "Each entry of 'solvent_combinations' must contain 2 or 3 solvents, "
                    f"got {combo!r}."
                )
        if (self.melting_temperature_C is None) != (self.melting_enthalpy_kJ_per_mol is None):
            raise ValueError(
                "Provide both melting_temperature_C and melting_enthalpy_kJ_per_mol, or neither."
            )
        return self


class SolventScreeningSubmissionConfig(SubmissionConfig):
    cores_per_task: int = Field(default=1, ge=1)
    max_time: str = Field(default="4hours")
    name: str = Field(default="QMA_SolvScreen")


class SolventScreeningInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for the solvent screening workflow."""

    command: Literal["solvent_screening"] = Field(default="solvent_screening")
    submission_config: SolventScreeningSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=SolventScreeningSubmissionConfig,
    )
    solvent_screening_options: SolventScreeningOptions = Field(
        default_factory=SolventScreeningOptions,
        description="Solvent screening workflow options.",
    )


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_config(argv: list[str] | None = None) -> SolventScreeningInterfaceConfig:
    parser = ArgumentParser(
        description=(
            "COSMOtherm solvent screening workflow — pure-solvent and binary/ternary "
            "mixture solubility calculations on previously generated .cosmo files."
        ),
        env_prefix="qm_atlas_SOLVENT_SCREENING_",
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to YAML config file")
    parser.add_class_arguments(SolventScreeningInterfaceConfig)

    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    return SolventScreeningInterfaceConfig(**interface_namespace)


# ---------------------------------------------------------------------------
# CSV append helper
# ---------------------------------------------------------------------------


def _references_to_json(references: list[ReferenceSolubilityConfig]) -> str:
    if not references:
        return ""
    return json.dumps([ref.model_dump() for ref in references], sort_keys=True)


def _append_results(
    cpd_dir: compound_dir.CpdDir,
    df: pd.DataFrame,
    *,
    state_names: list[str],
    calculation_type: str,
    mode: str,
    references_json: str,
    num_solvents: int,
) -> None:
    if df is None or df.empty:
        _logger.warning(
            f"Empty solvent-screening result for states {state_names} ({calculation_type})."
        )
        return

    try:
        annotated = cosmo_solubility.parse_solubility_data(df, num_solvents=num_solvents)
    except Exception as exc:  # pylint: disable=broad-except
        _logger.error(f"Failed to post-process solubility data ({calculation_type}): {exc}")
        _logger.error(traceback.format_exc())
        annotated = df.copy()

    annotated.insert(0, STATES_COL, ",".join(state_names))
    annotated.insert(1, CALC_TYPE_COL, calculation_type)
    annotated.insert(2, MODE_COL, mode)
    annotated.insert(3, REFERENCES_COL, references_json)

    cpd_dir.append_solubility_results(annotated)


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_local(config: SolventScreeningInterfaceConfig) -> None:
    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options: dict[str, Any] = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="solvent_screening")

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
                _logger.error("Got ^C while running solvent screening jobs.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"Solvent screening failed for {cpd_dir.cpd_name}: {exc}")
                _logger.error(traceback.format_exc())
            finally:
                eval_time = time.perf_counter() - start_time
                _logger.info(f"Evaluated solvent screening in {eval_time:.2f} seconds")
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
    config: SolventScreeningInterfaceConfig,
) -> None:
    _logger.info(f"Working on compound {cpd_dir.cpd_name}")
    options = config.solvent_screening_options

    if not options.solvent_names and not options.solvent_combinations:
        _logger.error(
            "No solvents configured for screening; set 'solvent_names' or 'solvent_combinations'."
        )
        return

    neutral_states, cosmo_files = _collect_neutral_cosmo_files(cpd_dir)

    if not neutral_states:
        _logger.error(
            f"No neutral (charge=0) states registered for {cpd_dir.cpd_name}; "
            "skipping solvent screening."
        )
        return

    if not cosmo_files:
        _logger.error(
            f"No COSMO files available for neutral states of {cpd_dir.cpd_name}; "
            "skipping solvent screening."
        )
        return

    _logger.info(
        f"Combining {len(cosmo_files)} neutral-state COSMO conformers "
        f"({len(neutral_states)} states: {neutral_states}) into one screening call."
    )

    references_dataclass = [ref.to_dataclass() for ref in options.references]
    references_json = _references_to_json(options.references)
    # COSMOtherm treats the calculation as "iterative" (absolute) when reference
    # solubilities or melting data are supplied; otherwise it runs in "relative" mode.
    has_anchor = bool(references_dataclass) or options.melting_temperature_C is not None
    mode = "absolute" if has_anchor else "relative"

    common_kwargs: dict[str, Any] = {
        "melting_temperature_C": options.melting_temperature_C,
        "melting_enthalpy_kJ_per_mol": options.melting_enthalpy_kJ_per_mol,
        "level": options.level,
        "references": references_dataclass if references_dataclass else None,
        "solvents_dirs": options.solvents_dirs,
        "temperature": options.temperature,
        "output_units": options.output_units,
        "use_config_cosmo_dir": options.use_config_cosmo_dir,
        "n_cores": config.n_cores or 1,
    }

    if options.solvent_names:
        _logger.info(
            f"Running pure-solvent screening for {cpd_dir.cpd_name} "
            f"over {len(options.solvent_names)} solvents."
        )
        try:
            df = cosmo_solubility.calculate_solubility(
                compound_conformers=list(cosmo_files),
                solvent_names=list(options.solvent_names),
                **common_kwargs,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(f"Pure-solvent screening failed for {cpd_dir.cpd_name}: {exc}")
            _logger.error(traceback.format_exc())
        else:
            _append_results(
                cpd_dir,
                df,
                state_names=neutral_states,
                calculation_type="pure",
                mode=mode,
                references_json=references_json,
                num_solvents=1,
            )

    if options.solvent_combinations:
        _logger.info(
            f"Running mixture screening for {cpd_dir.cpd_name} over "
            f"{len(options.solvent_combinations)} combinations."
        )
        try:
            df = cosmo_solubility.calculate_solubility_mixture(
                compound_conformers=list(cosmo_files),
                solvent_combinations=[list(c) for c in options.solvent_combinations],
                num_steps_binary=options.num_steps_binary,
                num_steps_ternary=options.num_steps_ternary,
                optimize=options.optimize,
                **common_kwargs,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(f"Mixture screening failed for {cpd_dir.cpd_name}: {exc}")
            _logger.error(traceback.format_exc())
        else:
            mixture_num_solvents = max(len(c) for c in options.solvent_combinations)
            _append_results(
                cpd_dir,
                df,
                state_names=neutral_states,
                calculation_type="mixture",
                mode=mode,
                references_json=references_json,
                num_solvents=mixture_num_solvents,
            )


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def submit(config: SolventScreeningInterfaceConfig) -> None:
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    worker_paths = extract_worker_paths(config.input, workflow_type="solvent_screening")
    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} molecules to process")
    submission.submit(worker_paths=worker_paths, config=config, command=COMMAND)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    config: SolventScreeningInterfaceConfig | None = None,
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
