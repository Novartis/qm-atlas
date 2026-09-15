"""COSMOtherm pKa calculation CLI.

Compute pKa values for registered states inside each compound directory using
COSMOtherm's :func:`qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_pka`.

Two modes are supported:

* ``macro`` (default) — group all registered states by formal charge and run
  one pKa calculation per adjacent charge pair, passing all cosmo files in
  each charge group as a Boltzmann-weighted ensemble to COSMOtherm.
* ``micro`` — enumerate all individual ``(parent_state, child_state)`` pairs
  with ``|Δq| == 1`` and run a separate pKa calculation for each pair.

Results are appended to the compound's ``pka.csv`` via
:meth:`CpdDir.add_pka_info`, with the ``Method`` column set to
``"cosmotherm_{level}_{mode}"`` so they can be distinguished from MoKa-derived
values.
"""

import logging
import sys
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Literal

from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    SubmissionConfig,
    apply_software_config,
    extract_worker_paths,
    setup_logging,
)
from qm_atlas.command_line.file_interface import compound_dir, task_files
from qm_atlas.command_line.file_interface.compound_dir import PkaInfo
from qm_atlas.command_line.utils import submission
from qm_atlas.wrappers.cosmotherm import cosmo_tasks

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas cosmo_pka"

CalculationMode = Literal["macro", "micro"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class CosmoPkaOptions(BaseModel):
    """Options for the cosmo_pka workflow."""

    mode: CalculationMode = Field(
        default="macro",
        description=(
            "'macro': one pKa per adjacent charge group (Boltzmann-weighted ensemble); "
            "'micro': one pKa per (parent_state, child_state) pair with |dq|==1."
        ),
    )
    level: str = Field(
        default="bp-tzvpd",
        description="COSMOtherm level of theory (bp-tzvpd, bp-tzvp, bp-svp, dmol3-pbe).",
    )
    solvent_name: str = Field(
        default="h2o",
        description="Solvent for the pKa calculation. Only solvents supported by "
        "COSMOtherm's pKa module are accepted (default: h2o).",
    )
    temperature_Celsius: float = Field(
        default=25.0,
        description="Temperature in Celsius.",
    )
    solvents_dirs: list[Path] = Field(
        default_factory=list,
        description="Extra directories to search for solvent .cosmo files.",
    )
    use_config_cosmo_dir: bool = Field(
        default=True,
        description="Also search the software-config-provided cosmo databases.",
    )
    reverse_form: bool = Field(
        default=False,
        description=(
            "Reverse the ACID/BASE form assignment. By default, charge pairs involving "
            "negative states (e.g. -1->0) use ACID form and pairs involving positive states "
            "(e.g. 0->+1) use BASE form. When True, these assignments are swapped."
        ),
    )


# ---------------------------------------------------------------------------
# Submission defaults
# ---------------------------------------------------------------------------


class CosmoPkaSubmissionConfig(SubmissionConfig):
    cores_per_task: int = Field(default=1, ge=1)
    max_time: str = Field(default="4hours")
    name: str = Field(default="QMA_CosmoPka")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class CosmoPkaInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for the cosmo_pka workflow."""

    command: Literal["cosmo_pka"] = Field(default="cosmo_pka")
    submission_config: CosmoPkaSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=CosmoPkaSubmissionConfig,
    )
    cosmo_pka_options: CosmoPkaOptions = Field(default_factory=CosmoPkaOptions)


def load_config(argv: list[str] | None = None) -> CosmoPkaInterfaceConfig:
    parser = ArgumentParser(
        description=(
            "Compute COSMOtherm pKa values across registered states in each compound directory."
        ),
        env_prefix="qm_atlas_COSMO_PKA_",
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to YAML config file")
    parser.add_class_arguments(CosmoPkaInterfaceConfig)
    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    return CosmoPkaInterfaceConfig(**interface_namespace)


# ---------------------------------------------------------------------------
# Per-compound helpers
# ---------------------------------------------------------------------------


def _collect_cosmo_files(cpd_dir: compound_dir.CpdDir, state_name: str) -> list[Path]:
    """Return the list of cosmo files that exist for a registered state."""
    cosmo_files: list[Path] = []
    for sdf_file in cpd_dir.registry_handler.get_result_files(state_name):
        cosmo_file = cpd_dir.get_associated_cosmo_file(sdf_file)
        if cosmo_file.exists():
            cosmo_files.append(cosmo_file)
        else:
            _logger.debug(f"Missing cosmo file for {sdf_file}; skipping that conformer.")
    return cosmo_files


def _charge_to_states(cpd_dir: compound_dir.CpdDir) -> dict[int, list[str]]:
    """Group all registered states by their formal charge."""
    registry = cpd_dir.registry_handler.get_registry()
    by_charge: dict[int, list[str]] = {}
    for name, entry in registry.items():
        by_charge.setdefault(entry.charge, []).append(name)
    return by_charge


def _states_with_cosmo(
    cpd_dir: compound_dir.CpdDir, state_names: list[str]
) -> list[tuple[str, list[Path]]]:
    """For each state name, collect existing cosmo files; drop states with none."""
    result: list[tuple[str, list[Path]]] = []
    for name in state_names:
        cosmo_files = _collect_cosmo_files(cpd_dir, name)
        if not cosmo_files:
            _logger.warning(f"No cosmo files available for state {name!r}; skipping.")
            continue
        result.append((name, cosmo_files))
    return result


def _get_form_and_roles(higher_q: int, lower_q: int, reverse_form: bool) -> tuple[str, int, int]:
    """Return (form, parent_charge, child_charge) for an adjacent charge pair.

    Default assignment:
    - Pairs where lower_q < 0 (e.g. -1->0, -2->-1): ACID; parent = higher-charge state.
    - Pairs where higher_q > 0 (e.g. 0->+1, +1->+2): BASE; parent = lower-charge state.

    When *reverse_form* is True the ACID/BASE assignment (and therefore parent/child
    roles) are swapped.
    """
    natural_form = "ACID" if lower_q < 0 else "BASE"
    form = ("BASE" if natural_form == "ACID" else "ACID") if reverse_form else natural_form
    if form == "ACID":
        # ACID: parent is the protonated (higher-charge) state
        return form, higher_q, lower_q
    else:
        # BASE: parent is the neutral/deprotonated (lower-charge) state
        return form, lower_q, higher_q


# ---------------------------------------------------------------------------
# pKa drivers
# ---------------------------------------------------------------------------


def _run_macro(
    cpd_dir: compound_dir.CpdDir,
    options: CosmoPkaOptions,
    method: str,
) -> int:
    """Run one pKa per adjacent charge group; returns number of values added."""
    by_charge = _charge_to_states(cpd_dir)
    if not by_charge:
        return 0

    charges_sorted = sorted(by_charge.keys())
    n_written = 0

    for higher_q in charges_sorted:
        lower_q = higher_q - 1
        if lower_q not in by_charge:
            continue

        form, parent_q, child_q = _get_form_and_roles(higher_q, lower_q, options.reverse_form)

        parent_states = _states_with_cosmo(cpd_dir, by_charge[parent_q])
        child_states = _states_with_cosmo(cpd_dir, by_charge[child_q])
        if not parent_states or not child_states:
            continue

        parent_names = [name for name, _ in parent_states]
        child_names = [name for name, _ in child_states]
        parent_cosmo = [path for _, paths in parent_states for path in paths]
        child_cosmo = [path for _, paths in child_states for path in paths]

        _logger.info(
            f"Macro pKa for charge {higher_q}->{lower_q} (form={form}): "
            f"parents={parent_names} ({len(parent_cosmo)} cosmo files), "
            f"children={child_names} ({len(child_cosmo)} cosmo files)"
        )
        try:
            pka_value = cosmo_tasks.calculate_pka(
                parent_conformers=parent_cosmo,
                child_conformers=child_cosmo,
                form=form,
                level=options.level,
                solvent_name=options.solvent_name,
                temperature_Celsius=options.temperature_Celsius,
                solvents_dirs=list(options.solvents_dirs) or None,
                use_config_cosmo_dir=options.use_config_cosmo_dir,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(
                f"calculate_pka failed for parents={parent_names} children={child_names}: {exc}"
            )
            _logger.error(traceback.format_exc())
            continue

        cpd_dir.add_pka_info(
            PkaInfo(
                pka_type=form,
                pka_value=float(pka_value),
                method=method,
                parent_states=parent_names,
                child_states=child_names,
            )
        )
        n_written += 1

    return n_written


def _run_micro(
    cpd_dir: compound_dir.CpdDir,
    options: CosmoPkaOptions,
    method: str,
) -> int:
    """Run one pKa per (parent, child) pair with charge diff = 1."""
    by_charge = _charge_to_states(cpd_dir)
    if not by_charge:
        return 0

    n_written = 0
    for higher_q in sorted(by_charge.keys()):
        lower_q = higher_q - 1
        if lower_q not in by_charge:
            continue

        form, parent_q, child_q = _get_form_and_roles(higher_q, lower_q, options.reverse_form)

        parents = _states_with_cosmo(cpd_dir, by_charge[parent_q])
        children = _states_with_cosmo(cpd_dir, by_charge[child_q])

        for (parent_name, parent_cosmo), (child_name, child_cosmo) in product(parents, children):
            _logger.info(
                f"Micro pKa for {parent_name} -> {child_name} (form={form}): "
                f"(parents: {len(parent_cosmo)} cosmo files, children: {len(child_cosmo)})"
            )
            try:
                pka_value = cosmo_tasks.calculate_pka(
                    parent_conformers=parent_cosmo,
                    child_conformers=child_cosmo,
                    form=form,
                    level=options.level,
                    solvent_name=options.solvent_name,
                    temperature_Celsius=options.temperature_Celsius,
                    solvents_dirs=list(options.solvents_dirs) or None,
                    use_config_cosmo_dir=options.use_config_cosmo_dir,
                )
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"calculate_pka failed for {parent_name} -> {child_name}: {exc}")
                _logger.error(traceback.format_exc())
                continue

            cpd_dir.add_pka_info(
                PkaInfo(
                    pka_type=form,
                    pka_value=float(pka_value),
                    method=method,
                    parent_states=[parent_name],
                    child_states=[child_name],
                )
            )
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Worker entry points
# ---------------------------------------------------------------------------


def _method_label(options: CosmoPkaOptions) -> str:
    """Generate method label for pKa results, including non-default parameters."""
    parts = [f"cosmotherm_{options.level}_{options.mode}"]

    # Add solvent if not default (h2o and water are equivalent)
    if options.solvent_name.lower() not in ("h2o", "water"):
        parts.append(options.solvent_name)

    # Add temperature if not default
    if options.temperature_Celsius != 25.0:
        parts.append(f"TC={options.temperature_Celsius}")

    return "_".join(parts)


def run_for_cpd(cpd_dir: compound_dir.CpdDir, config: CosmoPkaInterfaceConfig) -> None:
    """Run pKa calculations for a single compound directory."""
    options = config.cosmo_pka_options
    method = _method_label(options)
    _logger.info(
        f"Computing COSMOtherm pKa for {cpd_dir.cpd_name} (mode={options.mode}, method={method})"
    )
    if options.mode == "macro":
        n = _run_macro(cpd_dir, options, method)
    else:
        n = _run_micro(cpd_dir, options, method)
    _logger.info(f"Wrote {n} pKa value(s) for {cpd_dir.cpd_name}.")


def run_local(config: CosmoPkaInterfaceConfig) -> None:
    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options = {
        "filemode": "a",
        "level": logging.DEBUG if config.verbose else logging.INFO,
        "format": logging_fmt,
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    worker_paths = extract_worker_paths(config.input, workflow_type="cosmotherm_properties")

    # pKa needs all states of a compound together; iterate unique compound dirs only.
    seen_cpd_dirs: set[Path] = set()
    prev_log_file = None
    try:
        for cpd_dir_obj, _sdf_file, log_file in worker_paths:
            if log_file != prev_log_file:
                task_files.change_log_file(log_file, **logging_options)
                prev_log_file = log_file
            if cpd_dir_obj.dir in seen_cpd_dirs:
                continue
            seen_cpd_dirs.add(cpd_dir_obj.dir)

            start_time = time.perf_counter()
            try:
                run_for_cpd(cpd_dir_obj, config)
            except KeyboardInterrupt:
                _logger.error("Got ^C while running cosmo_pka jobs.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"cosmo_pka failed for {cpd_dir_obj.cpd_name}: {exc}")
                _logger.error(traceback.format_exc())
            finally:
                _logger.info(f"Evaluated cosmo_pka in {time.perf_counter() - start_time:.2f} s")
    finally:
        task_files.close_task_log_handler()


def submit(config: CosmoPkaInterfaceConfig) -> None:
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    worker_paths = extract_worker_paths(config.input, workflow_type="cosmotherm_properties")
    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} state(s) (jobs will be deduplicated per compound)")
    submission.submit(worker_paths=worker_paths, config=config, command=COMMAND)


def main(
    argv: list[str] | None = None,
    config: CosmoPkaInterfaceConfig | None = None,
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
