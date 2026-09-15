"""Generate protonation states CLI.

For each existing input state in each compound directory, run a configurable
protonation backend and register the surviving protomers as additional input
states. Protomer naming follows the ``{parent_state}{charge_suffix}{n}``
convention used elsewhere in qm_atlas (see
:func:`qm_atlas.wrappers.moka.process.get_charge_suffix`); collisions are
resolved by appending an increasing integer.

Three backends are exposed:

* ``moka_blabber`` - MoKa's ``blabber_sd`` tool. No pKa info, but reports
  per-protomer abundances.
* ``moka_ionic_species`` - MoKa pKa prediction + sequential ionic-species
  enumeration. Reports pKa values and atom indices for each transition.
* ``moka_single_charge`` - MoKa pKa prediction + singly-charged ion
  enumeration (one ion per ionisable centre).

Whenever the backend reports pKa information, it is appended to the
compound's ``pka.csv`` via :meth:`CpdDir.add_pka_info` with method
``"{backend_name}"`` and the ionisation atom index in the ``Atom_Index``
column.
"""

import logging
import sys
import time
import traceback
from collections import Counter
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
from qm_atlas.command_line.file_interface.compound_dir import PkaInfo
from qm_atlas.command_line.utils import submission
from qm_atlas.tasks import protonation
from qm_atlas.wrappers.moka.process import get_charge_suffix

_logger = logging.getLogger(__name__)

COMMAND = "qm-atlas protonate"


class ProtonateOptions(BaseModel):
    """Options for the protonate workflow."""

    # Optional[union] with a None default lets jsonargparse resolve a single
    # discriminated-union field from YAML without merging a concrete default's
    # fields; run_job falls back to MokaBlabberConfig when left unset.
    backend: protonation.ProtonationConfig | None = Field(
        default=None,
        description=(
            "Protonation backend configuration. Available backends: "
            "'moka_blabber' (default), 'moka_ionic_species', 'moka_single_charge'."
        ),
    )

    @field_validator("backend", mode="before")
    @classmethod
    def _deserialize_backend(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return protonation.deserialize_protonation_config(value)
        return value


class ProtonateSubmissionConfig(SubmissionConfig):
    cores_per_task: int = Field(default=1, ge=1)
    max_time: str = Field(default="4hours")
    name: str = Field(default="QMA_PROTONATE")


class ProtonateInterfaceConfig(BaseInterfaceConfig):
    """Complete configuration for the protonate workflow."""

    command: Literal["protonate"] = Field(default="protonate")
    submission_config: ProtonateSubmissionConfig = Field(  # type: ignore[assignment]
        default_factory=ProtonateSubmissionConfig,
    )
    protonate_options: ProtonateOptions = Field(default_factory=ProtonateOptions)


def load_config(argv: list[str] | None = None) -> ProtonateInterfaceConfig:
    parser = ArgumentParser(
        description=(
            "Generate protonation states with a configurable backend "
            "(moka_blabber, moka_ionic_species, moka_single_charge) and register "
            "surviving protomers as new input states."
        ),
        env_prefix="qm_atlas_PROTONATE_",
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to YAML config file")
    parser.add_class_arguments(ProtonateInterfaceConfig)
    args = parser.parse_args(argv)
    interface_namespace = parser.instantiate_classes(args)
    return ProtonateInterfaceConfig(**interface_namespace)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)


def _next_free_protomer_name(
    parent_state_name: str,
    suffix: str,
    suffix_counts: Counter,
    registered: set[str],
) -> str:
    """Return the next free ``{parent}{suffix}[n]`` name.

    The first occurrence of *suffix* under *parent_state_name* is bare
    (``{parent}{suffix}``); subsequent ones get an increasing integer
    appended (``{parent}{suffix}2``, ``{parent}{suffix}3``, ...), matching
    the convention used by :class:`qm_atlas.pka.PkaCalculationInfo`.
    """
    while True:
        suffix_counts[suffix] += 1
        count = suffix_counts[suffix]
        candidate = (
            f"{parent_state_name}{suffix}" if count == 1 else f"{parent_state_name}{suffix}{count}"
        )
        if candidate not in registered:
            return candidate


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def run_job(
    cpd_dir: compound_dir.CpdDir,
    sdf_file: Path,
    config: ProtonateInterfaceConfig,
) -> None:
    _logger.info(f"Generating protomers for {sdf_file.stem}")

    mols = input_file.read_input_sdf(sdf_file)
    if len(mols) == 0:
        _logger.warning(f"No molecules read from {sdf_file}; skipping.")
        return
    if len(mols) > 1:
        _logger.warning(f"{sdf_file} contains more than one molecule; using the first only.")
    parent_mol = mols[0]
    parent_state_name = cpd_dir.find_state_by_file(sdf_file)
    parent_smiles = _canonical_smiles(parent_mol)

    backend_config = config.protonate_options.backend or protonation.MokaBlabberConfig()
    result = protonation.run_protonation(
        parent_mol,
        backend_config,
        scr=cpd_dir.get_trace_dir(),
    )

    if not result.protomers and not result.transitions:
        _logger.info(f"Backend produced no protomers for {parent_state_name}.")
        return

    smiles_to_name = _register_protomers(
        cpd_dir=cpd_dir,
        result=result,
        parent_state_name=parent_state_name,
        parent_smiles=parent_smiles,
        backend_name=backend_config.backend,
    )

    _register_transitions(
        cpd_dir=cpd_dir,
        result=result,
        smiles_to_name=smiles_to_name,
        backend_name=backend_config.backend,
    )


def _register_protomers(
    cpd_dir: compound_dir.CpdDir,
    result: protonation.ProtonationResult,
    parent_state_name: str,
    parent_smiles: str,
    backend_name: str,
) -> dict[str, str]:
    """Register each new protomer; return a SMILES -> state-name map.

    The map always contains the parent SMILES so transitions involving the
    parent can resolve to a registered name.
    """
    existing_smiles = set(cpd_dir.get_registered_smiles())
    registered_names = set(cpd_dir.registry_handler.get_registered_names())
    suffix_counts: Counter = Counter()

    smiles_to_name: dict[str, str] = {}
    # seed with all existing registered (smiles -> name) so transitions
    # involving already-known states resolve immediately.
    for name in registered_names:
        entry = cpd_dir.registry_handler.get_entry(name)
        smiles_to_name[entry.smiles] = name
    smiles_to_name.setdefault(parent_smiles, parent_state_name)

    n_duplicates = 0
    n_new = 0
    for entry in result.protomers:
        protomer = entry.mol
        smi = _canonical_smiles(protomer)
        if smi in existing_smiles:
            n_duplicates += 1
            smiles_to_name.setdefault(smi, _find_name_for_smiles(cpd_dir, smi))
            continue

        suffix = get_charge_suffix(protomer)
        new_name = _next_free_protomer_name(
            parent_state_name, suffix, suffix_counts, registered_names
        )
        protomer.SetProp("_Name", new_name)
        abundance_str = f" (abundance={entry.abundance})" if entry.abundance is not None else ""
        description = f"protomer of {parent_state_name} via {backend_name}{abundance_str}"
        try:
            cpd_dir.add_input_structure(protomer, overwrite=False, description=description)
        except (OSError, ValueError) as exc:
            _logger.error(f"Failed to register protomer {new_name}: {exc}")
            continue

        existing_smiles.add(smi)
        registered_names.add(new_name)
        smiles_to_name[smi] = new_name
        n_new += 1

    _logger.info(
        f"Protonation for {parent_state_name} ({backend_name}): "
        f"{len(result.protomers)} protomers returned, "
        f"{n_new} registered as new states, {n_duplicates} skipped as duplicates."
    )
    return smiles_to_name


def _find_name_for_smiles(cpd_dir: compound_dir.CpdDir, smiles: str) -> str:
    try:
        return cpd_dir.registry_handler.find_name_by_smiles(smiles)
    except ValueError:
        return ""


def _register_transitions(
    cpd_dir: compound_dir.CpdDir,
    result: protonation.ProtonationResult,
    smiles_to_name: dict[str, str],
    backend_name: str,
) -> None:
    """Persist each backend-reported transition to pka.csv."""
    if not result.transitions:
        return
    n_written = 0
    for transition in result.transitions:
        parent_name = smiles_to_name.get(transition.parent_smiles)
        child_name = smiles_to_name.get(transition.child_smiles)
        if not parent_name or not child_name:
            _logger.warning(
                "Skipping transition pKa=%.2f: could not map both SMILES to a "
                "registered state (parent=%r, child=%r).",
                transition.pka_value,
                transition.parent_smiles,
                transition.child_smiles,
            )
            continue
        cpd_dir.add_pka_info(
            PkaInfo(
                pka_type=transition.pka_type,
                pka_value=transition.pka_value,
                method=backend_name,
                parent_states=[parent_name],
                child_states=[child_name],
                atom_index=transition.atom_index,
            )
        )
        n_written += 1
    _logger.info(f"Wrote {n_written} pKa transition(s) from {backend_name} to pka.csv.")


def run_local(config: ProtonateInterfaceConfig) -> None:
    logging_fmt = "%(asctime)s, %(name)s, %(levelname)s: %(message)s (%(filename)s:%(lineno)s)"
    logging_options: dict[str, Any] = {
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
                _logger.error("Got ^C while running protonation jobs.")
                sys.exit()
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(f"protonate failed for {sdf_file}: {exc}")
                _logger.error(traceback.format_exc())
            finally:
                _logger.info(f"Evaluated protonate in {time.perf_counter() - start_time:.2f} s")
    finally:
        task_files.close_task_log_handler()


def submit(config: ProtonateInterfaceConfig) -> None:
    _logger.info("Preparing UGE job submission")
    _logger.info(f"Job name: {config.submission_config.name}")
    worker_paths = extract_worker_paths(config.input, workflow_type="conformer_expansion")
    if not worker_paths:
        raise ValueError("No molecules found to process")
    _logger.info(f"Found {len(worker_paths)} input states to process")
    submission.submit(worker_paths=worker_paths, config=config, command=COMMAND)


def main(
    argv: list[str] | None = None,
    config: ProtonateInterfaceConfig | None = None,
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
