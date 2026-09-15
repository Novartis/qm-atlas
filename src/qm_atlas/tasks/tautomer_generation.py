"""Tautomer generation and selection tasks.

Covers three pipeline steps:

1. :func:`generate_tautomers` — run unicon, add the original molecule, deduplicate.
2. :func:`generate_tautomer_conformers` — run fast conformer generation for each tautomer.
3. :func:`select_stable_tautomers` — filter by XTB energy threshold, strip conformers,
   remove explicit hydrogens.
"""

import logging
from pathlib import Path

import numpy as np
from ppqm import chembridge
from rdkit import Chem

from qm_atlas import units
from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.tasks import calculate_properties
from qm_atlas.tasks.calculate_properties import XtbSinglePointOptions
from qm_atlas.tasks.conformer_generation import (
    ConformerGenerationOptions,
    OmegaOptions,
    RdkitOptions,
)
from qm_atlas.workflows.fast_conformers import generate_fast_conformers
from qm_atlas.wrappers import unicon
from qm_atlas.wrappers.xtb import XTB_ENERGY_KEY

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default options
# ---------------------------------------------------------------------------

TAUTOMER_CONFORMER_OPTIONS = ConformerGenerationOptions(
    conf_gens=[
        OmegaOptions(
            rms_threshold=0.5,
            max_conformers=5,
            start_from_corina=True,
        ),
        RdkitOptions(rms_threshold=0.5, max_conformers=5, method="KDG"),
    ],
    combination_name="fallback",
)

XTB_SP_CONFIG = XtbSinglePointOptions(
    solvation_model="alpb",
    solvent="water",
    calculate_fukui=False,
    xtb_command_add=list(),
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def generate_tautomers(molobj: Chem.Mol, scr: Path) -> list[Chem.Mol]:
    """Generate tautomers for a molecule using unicon.

    Runs unicon, appends the original molecule, then deduplicates by canonical
    SMILES so the returned list contains only structurally distinct tautomers.

    Args:
        molobj: Input molecule (should already be neutralized).
        scr: Scratch directory for unicon intermediate files.

    Returns:
        List of unique tautomer Mol objects (always includes the original).

    Raises:
        ValueError: If the deduplicated tautomer list is empty.
    """
    tautomers = unicon.get_tautomers(molobj, scr=scr)
    _logger.info(f"unicon found {len(tautomers)} tautomer(s)")

    if len(tautomers) == 0:
        _logger.warning("unicon found zero tautomers")

    tautomers.append(molobj)

    smis = {chembridge.molobj_to_smiles(t) for t in tautomers}
    tautomers: list[Chem.Mol | None] = [chembridge.smiles_to_molobj(s) for s in smis]
    _tautomers = [t for t in tautomers if t is not None]

    if not tautomers:
        raise ValueError("No tautomers could be generated")

    _logger.info(f"found {len(tautomers)} unique tautomer(s) after deduplication")
    return _tautomers


def generate_tautomer_conformers(
    tautomers: list[Chem.Mol],
    options: ConformerGenerationOptions = TAUTOMER_CONFORMER_OPTIONS,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    show_progress: bool = True,
) -> list[Chem.Mol]:
    """Generate fast conformers for each tautomer.

    Calls :func:`~qm_atlas.workflows.fast_conformers.generate_fast_conformers`
    on every tautomer in the list.

    Args:
        tautomers: List of tautomer Mol objects (no conformers required).
        options: Conformer generation options. Defaults to
            :data:`TAUTOMER_CONFORMER_OPTIONS` (Omega → RDKit fallback, max 5 confs).
        scr: Scratch directory for xTB calculations.
        n_cores: Number of CPU cores.
        show_progress: Show tqdm progress bar during optimization.

    Returns:
        List of tautomers with conformers embedded (same order as input).

    Raises:
        ValueError: If any tautomer produces zero conformers.
    """

    result: list[Chem.Mol] = []
    for i, tautomer in enumerate(tautomers):
        tautomer_3d = generate_fast_conformers(
            tautomer,
            conformer_generation_options=options,
            scr=scr,
            n_cores=n_cores,
            show_progress=show_progress,
        )
        if tautomer_3d.GetNumConformers() == 0:
            raise ValueError(f"Could not generate conformers for tautomer {i}")
        result.append(tautomer_3d)
    return result


def select_stable_tautomers(
    tautomers: list[Chem.Mol],
    xtb_sp_config: XtbSinglePointOptions = XTB_SP_CONFIG,
    threshold_kcal_mol: float = 7.0,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    show_progress: bool = True,
) -> list[Chem.Mol]:
    """Filter tautomers by XTB energy, keeping those within the energy threshold.

    Runs XTB single-point calculations on the conformers of each tautomer, takes
    the minimum energy per tautomer, converts to kcal/mol relative to the lowest
    tautomer, and keeps those below ``threshold_kcal_mol``.

    After filtering, conformers are stripped and explicit hydrogens removed so the
    returned molecules are clean graph-level objects ready for the next step.

    Args:
        tautomers: Tautomers with conformers (output of
            :func:`generate_tautomer_conformers`).
        xtb_sp_config: XTB single-point options for tautomer screening.
        threshold_kcal_mol: Energy window above the lowest tautomer (kcal/mol).
        n_cores: Number of CPU cores.
        scr: Scratch directory for xTB.
        show_progress: Show tqdm progress bar.

    Returns:
        Filtered list of tautomers as graph-level Mol objects (no conformers,
        no explicit Hs).
    """
    xtb_sp_results = calculate_properties.calculate_property_mols(
        tautomers,
        config=xtb_sp_config,
        n_cores=n_cores,
        scr=scr,
        show_progress=show_progress,
    )

    xtb_property_names = xtb_sp_config.get_property_names()
    energy_key = xtb_property_names[XTB_ENERGY_KEY]

    energies: list[list[float]] = []
    for result_lst in xtb_sp_results:
        sub_list: list[float] = []
        for result in result_lst:
            _logger.debug(f"XTB result: {result}")
            if result is None or energy_key not in result:
                energy = float("inf")
            else:
                energy = result[energy_key].get_property_value()
            sub_list.append(energy)
        energies.append(sub_list)

    graph_energies = np.array([np.min(e) for e in energies]) * units.hartree_to_kcalmol
    graph_energies -= np.min(graph_energies)

    (stable_indices,) = np.where(graph_energies < threshold_kcal_mol)
    stable = [tautomers[i] for i in stable_indices]

    # Strip conformers and remove Hs for clean graph representation
    stable = [chembridge.copy_molobj(mol) for mol in stable]
    stable = [Chem.RemoveHs(mol) for mol in stable]

    _logger.info(
        f"filtered to {len(stable)} stable tautomer(s) "
        f"(threshold: {threshold_kcal_mol} kcal/mol)"
    )
    return stable
