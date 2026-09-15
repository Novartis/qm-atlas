"""Protonation-state generation task.

This module exposes a pluggable backend registry for generating protonation
states. Three backends are currently implemented:

* ``moka_blabber`` wraps MoKa's ``blabber_sd`` tool and yields the most
  abundant protomers at a given pH. It reports per-protomer abundances but
  no pKa values or atom indices.
* ``moka_ionic_species`` uses MoKa's ``moka_cli`` pKa prediction together
  with :func:`qm_atlas.wrappers.moka.process.generate_ionic_species` to
  enumerate the sequentially ionised species across a pH window.
* ``moka_single_charge`` uses MoKa's pKa prediction together with
  :func:`qm_atlas.wrappers.moka.process.generate_single_charge_ions` to
  enumerate singly-charged species (one per ionisable centre).

All backends return a :class:`ProtonationResult` containing
:class:`ProtomerEntry` items (per-protomer mol + optional abundance) and
:class:`ProtonationPkaTransition` items (per-transition pKa / atom index).
Backends that do not provide a piece of information leave the corresponding
field as ``None`` (or the transition list empty).
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

from qm_atlas.wrappers.moka import blabber, moka, process

_logger = logging.getLogger(__name__)

ABUNDANCE_PROP = blabber.COLUMN_ABUNDANCE

# Canonical pKa-type labels used in the unified result and in pka.csv.
PKA_TYPE_ACID = "ACID"
PKA_TYPE_BASE = "BASE"

_MOKA_TO_LABEL = {moka.PkaTypes.ACIDIC: PKA_TYPE_ACID, moka.PkaTypes.BASIC: PKA_TYPE_BASE}


# ---------------------------------------------------------------------------
# Unified result types
# ---------------------------------------------------------------------------


@dataclass
class ProtomerEntry:
    """A single protomer produced by a protonation backend."""

    mol: Chem.Mol
    abundance: float | None = None


@dataclass
class ProtonationPkaTransition:
    """A pKa transition reported by a protonation backend.

    Protomers are referred to by canonical isomeric SMILES so callers can
    map them onto registered state names regardless of mol identity.
    """

    parent_smiles: str
    child_smiles: str
    pka_value: float
    pka_type: str
    atom_index: int | None = None


@dataclass
class ProtonationResult:
    """Unified return value of :func:`run_protonation`."""

    protomers: list[ProtomerEntry] = field(default_factory=list)
    transitions: list[ProtonationPkaTransition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backend configs
# ---------------------------------------------------------------------------


class ProtonationOptions(BaseModel):
    """Base class for protonation backends.

    Each concrete backend declares a unique ``backend`` literal that acts as the
    discriminator for (de)serialization within :data:`ProtonationConfig`.
    """

    # Reject unknown keys so we don't silently ignore typos in config files.
    model_config = ConfigDict(extra="forbid")


class MokaBlabberConfig(ProtonationOptions):
    """Generate protonation states with MoKa's ``blabber_sd`` tool."""

    backend: Literal["moka_blabber"] = Field(default="moka_blabber")
    ph: str = Field(
        default="7.4",
        description=(
            "pH passed to blabber's --pH flag. Accepts a single value, a range "
            "(e.g. '6-10'), or a comma-separated list (e.g. '6,7-10')."
        ),
    )
    abundance_threshold: float | None = Field(
        default=None,
        description=(
            "Only protomers with an abundance above this threshold (percent) are "
            "kept. Forwarded to blabber's -t flag when set; further filtering is "
            "applied on the returned ABUNDANCE property."
        ),
    )
    remove_zero_charged: bool = Field(
        default=False,
        description="If True, drop neutral protomers from the blabber output.",
    )


class _MokaPkaConfigBase(ProtonationOptions):
    """Common options for MoKa pKa-driven backends."""

    lower_ph_threshold: float | None = Field(
        default=None,
        description=(
            "Lower pH limit. Basic pKa centres below this value are dropped; "
            "if ``None`` no lower cutoff is applied."
        ),
    )
    upper_ph_threshold: float | None = Field(
        default=None,
        description=(
            "Upper pH limit. Acidic pKa centres above this value are dropped; "
            "if ``None`` no upper cutoff is applied."
        ),
    )


class MokaIonicSpeciesConfig(_MokaPkaConfigBase):
    """Enumerate ionic species using MoKa pKa prediction.

    Uses :func:`qm_atlas.wrappers.moka.process.generate_ionic_species`.
    """

    backend: Literal["moka_ionic_species"] = Field(default="moka_ionic_species")


class MokaSingleChargeConfig(_MokaPkaConfigBase):
    """Enumerate singly-charged ions using MoKa pKa prediction.

    Uses :func:`qm_atlas.wrappers.moka.process.generate_single_charge_ions`.
    """

    backend: Literal["moka_single_charge"] = Field(default="moka_single_charge")


PROTONATION_REGISTRY: dict[str, type[ProtonationOptions]] = {
    "moka_blabber": MokaBlabberConfig,
    "moka_ionic_species": MokaIonicSpeciesConfig,
    "moka_single_charge": MokaSingleChargeConfig,
}


# Discriminated union of all protonation config types. The ``backend`` literal
# field acts as the discriminator so Pydantic selects the concrete class
# automatically when deserializing from a dict, and serializes all fields during
# model_dump().
ProtonationConfig = Annotated[
    MokaBlabberConfig | MokaIonicSpeciesConfig | MokaSingleChargeConfig,
    Field(discriminator="backend"),
]


def deserialize_protonation_config(data: Any) -> ProtonationOptions:
    """Deserialize a protonation config from a dict using its ``backend`` field."""
    if isinstance(data, ProtonationOptions):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Cannot deserialize protonation config from {type(data)!r}")

    config_name = data.get("backend", "moka_blabber")
    if config_name not in PROTONATION_REGISTRY:
        raise ValueError(f"Unknown protonation backend: {config_name!r}")
    return PROTONATION_REGISTRY[config_name](**data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)


def _filter_by_abundance(
    entries: list[ProtomerEntry], threshold: float | None
) -> list[ProtomerEntry]:
    if threshold is None:
        return entries
    kept: list[ProtomerEntry] = []
    for entry in entries:
        if entry.abundance is None:
            _logger.debug("Protomer has no abundance; keeping it.")
            kept.append(entry)
            continue
        if entry.abundance >= threshold:
            kept.append(entry)
    return kept


def _predicted_pkas_to_centers(
    predicted_pkas: list[moka.PredictedPKA],
) -> list[process.PkaCenter]:
    """Convert MoKa ``PredictedPKA`` results to :class:`process.PkaCenter` objects.

    PKa centres without a localised atom index are dropped: the
    ``generate_*`` helpers cannot use them.
    """
    centers: list[process.PkaCenter] = []
    for pka in predicted_pkas:
        if pka.atom_index is None:
            _logger.warning(
                "Skipping MoKa pKa value %s (type=%s) without a localised atom index.",
                pka.value,
                pka.pka_type,
            )
            continue
        centers.append(
            process.PkaCenter(
                center_type=pka.pka_type,
                index=int(pka.atom_index),
                pka_value=float(pka.value),
            )
        )
    return centers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_protonation(
    mol: Chem.Mol,
    config: ProtonationOptions,
    scr: Path,
    keep_files: bool = False,
) -> ProtonationResult:
    """Run a protonation backend on *mol* and return the unified result.

    Args:
        mol: Parent molecule.
        config: Backend-specific configuration (a concrete :class:`ProtonationOptions` subclass).
        scr: Scratch directory for backend intermediates.
        keep_files: Whether to keep the backend intermediate files. Used for
            debugging. Defaults to False.

    Returns:
        :class:`ProtonationResult` with the protomers produced by the
        backend and, where supported, the pKa transitions linking them.
    """
    if isinstance(config, MokaBlabberConfig):
        return _run_moka_blabber(mol, config, scr, keep_files=keep_files)
    if isinstance(config, MokaIonicSpeciesConfig):
        return _run_moka_pka(mol, config, scr, mode="ionic_species", keep_files=keep_files)
    if isinstance(config, MokaSingleChargeConfig):
        return _run_moka_pka(mol, config, scr, mode="single_charge", keep_files=keep_files)
    raise ValueError(
        f"Unsupported protonation backend: {getattr(config, 'backend', type(config).__name__)!r}"
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _run_moka_blabber(
    mol: Chem.Mol,
    config: MokaBlabberConfig,
    scr: Path,
    keep_files: bool = False,
) -> ProtonationResult:
    protomers = blabber.get_protonation(
        mol,
        scr=scr,
        remove_zero_charged=config.remove_zero_charged,
        ph=config.ph,
        threshold=config.abundance_threshold,
        keep_files=keep_files,
    )
    entries: list[ProtomerEntry] = []
    for protomer in protomers:
        abundance: float | None = None
        if protomer.HasProp(ABUNDANCE_PROP):
            try:
                abundance = float(protomer.GetProp(ABUNDANCE_PROP))
            except (TypeError, ValueError):
                _logger.debug(
                    "Could not parse abundance %r; treating as missing.",
                    protomer.GetProp(ABUNDANCE_PROP),
                )
                abundance = None
        entries.append(ProtomerEntry(mol=protomer, abundance=abundance))
    entries = _filter_by_abundance(entries, config.abundance_threshold)
    return ProtonationResult(protomers=entries, transitions=[])


def _run_moka_pka(
    mol: Chem.Mol,
    config: _MokaPkaConfigBase,
    scr: Path,
    mode: str,
    keep_files: bool = False,
) -> ProtonationResult:
    """Shared driver for the two MoKa-pKa backends."""
    prediction = moka.get_moka_prediction(
        mol,
        scr=scr,
        lower_ph_threshold=config.lower_ph_threshold,
        upper_ph_threshold=config.upper_ph_threshold,
        keep_files=keep_files,
    )
    centers = _predicted_pkas_to_centers(prediction.predicted_pkas)
    if not centers:
        _logger.info("MoKa returned no usable pKa centres for the provided molecule.")
        return ProtonationResult(protomers=[], transitions=[])

    lower = config.lower_ph_threshold if config.lower_ph_threshold is not None else -math.inf
    upper = config.upper_ph_threshold if config.upper_ph_threshold is not None else math.inf

    if mode == "ionic_species":
        calc_info = process.generate_ionic_species(
            mol,
            centers,
            lower_ph_threshold=lower,
            upper_ph_threshold=upper,
        )
    elif mode == "single_charge":
        calc_info = process.generate_single_charge_ions(
            mol,
            centers,
            lower_ph_threshold=lower,
            upper_ph_threshold=upper,
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown MoKa-pKa mode: {mode!r}")

    return _calc_info_to_result(calc_info)


def _calc_info_to_result(calc_info: process.PkaCalculationInfo) -> ProtonationResult:
    """Convert a :class:`process.PkaCalculationInfo` to :class:`ProtonationResult`."""
    name_to_smiles: dict[str, str] = {}
    protomers: list[ProtomerEntry] = []
    for name, (m, _expand) in calc_info.name_to_mol.items():
        smi = _canonical_smiles(m)
        name_to_smiles[name] = smi
        protomers.append(ProtomerEntry(mol=m, abundance=None))

    transitions: list[ProtonationPkaTransition] = []
    for center in calc_info.pka_centers.values():
        if center.parent_name is None or center.child_name is None:
            continue
        parent_smi = name_to_smiles.get(center.parent_name)
        child_smi = name_to_smiles.get(center.child_name)
        if parent_smi is None or child_smi is None:
            continue
        transitions.append(
            ProtonationPkaTransition(
                parent_smiles=parent_smi,
                child_smiles=child_smi,
                pka_value=float(center.pka_value),
                pka_type=_MOKA_TO_LABEL.get(center.center_type, center.center_type),
                atom_index=int(center.index),
            )
        )

    return ProtonationResult(protomers=protomers, transitions=transitions)
