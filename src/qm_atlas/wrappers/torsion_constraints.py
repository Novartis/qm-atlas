"""Heavy-atom torsion selection for constrained xtb optimizations.

The module aims to identify torsion angles involving only heavy atoms that are
suitable to fix in constrained optimizations. It follows the logic used by xtb's
``constrain_all_torsions`` routine, but restricts to heavy atoms only. With the
atom indices found, it writes the resulting torsions as explicit ``dihedral: i,j,k,l,auto``
restraints.  Like xtb it keeps a bond ``j-k`` when both central atoms have a
coordination number of at least 2 (i.e. carry a substituent besides each other) and
skips near-linear arrangements.
"""

import logging
from collections.abc import Sequence
from itertools import product

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

_logger = logging.getLogger(__name__)

#: Angles this close to 180° (in degrees) make a torsion ill-defined, so torsions
#: containing them are skipped.  Matches the 0.2 rad threshold xtb uses in
#: ``constrain_all_torsions``.
LINEAR_ANGLE_TOLERANCE_DEG = 11.46


def find_heavy_atom_torsions(
    mol: Chem.Mol,
    conf_id: int = -1,
    linear_angle_tolerance_deg: float = LINEAR_ANGLE_TOLERANCE_DEG,
) -> list[tuple[int, int, int, int]]:
    """Select one heavy-atom torsion per bond that carries a torsional degree of freedom.

    Follows xtb's ``constrain_all_torsions`` logic: a bond ``j-k`` is kept when both
    central atoms have a heavy-atom coordination number of at least 2 (i.e. each has a
    heavy neighbour besides the other), and torsions with a near-linear bond angle are
    skipped. Unlike xtb, only heavy atoms are counted and used.

    Args:
        mol (Chem.Mol):
            Molecule with connectivity and at least one conformation. The
            conformation is only used to reject near-linear arrangements.
        conf_id (int, optional):
            Conformation used for the near-linearity check. Defaults to -1
            (the first/only conformation).
        linear_angle_tolerance_deg (float, optional):
            Torsions containing a bond angle within this many degrees of 180°
            are skipped, because the torsion is then ill-defined. Defaults to
            :data:`LINEAR_ANGLE_TOLERANCE_DEG`.

    Raises:
        ValueError: If the molecule has no conformation.

    Returns:
        list[tuple[int, int, int, int]]: One (i, j, k, l) tuple of 0-based
        atom indices per selected bond, ordered by the central bond's atom
        indices. j and k are the central bond's atoms.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Torsion constraints require a molecule with a conformation to check "
            "for near-linear arrangements."
        )

    conformer = mol.GetConformer(conf_id)
    # Heavy-atom coordination numbers: the sorted heavy neighbours of each heavy atom.
    heavy_neighbors = {
        atom.GetIdx(): sorted(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1
        )
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    }

    torsions: list[tuple[int, int, int, int]] = []
    for bond in mol.GetBonds():
        central_j, central_k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if central_j not in heavy_neighbors or central_k not in heavy_neighbors:
            continue
        # xtb's coordination-number gate (bond(j,j) >= 2 and bond(k,k) >= 2), i.e. both
        # central atoms need a heavy substituent besides each other to define a torsion.
        if len(heavy_neighbors[central_j]) < 2 or len(heavy_neighbors[central_k]) < 2:
            continue

        torsion = _first_well_defined_torsion(
            conformer,
            [idx for idx in heavy_neighbors[central_j] if idx != central_k],
            central_j,
            central_k,
            [idx for idx in heavy_neighbors[central_k] if idx != central_j],
            linear_angle_tolerance_deg,
        )
        if torsion is None:
            _logger.debug(
                f"No well-defined torsion found for bond {central_j}-{central_k}; skipping it."
            )
            continue
        torsions.append(torsion)

    return sorted(torsions, key=lambda atoms: (atoms[1], atoms[2]))


def _first_well_defined_torsion(
    conformer: Chem.Conformer,
    candidates_i: Sequence[int],
    central_j: int,
    central_k: int,
    candidates_l: Sequence[int],
    linear_angle_tolerance_deg: float,
) -> tuple[int, int, int, int] | None:
    """Return the highest-priority torsion whose bond angles are not near-linear."""
    for idx_i, idx_l in product(candidates_i, candidates_l):
        if idx_i == idx_l:
            continue
        angle_ijk = rdMolTransforms.GetAngleDeg(conformer, idx_i, central_j, central_k)
        angle_jkl = rdMolTransforms.GetAngleDeg(conformer, central_j, central_k, idx_l)
        if (
            abs(180.0 - angle_ijk) < linear_angle_tolerance_deg
            or abs(180.0 - angle_jkl) < linear_angle_tolerance_deg
        ):
            continue
        return (idx_i, central_j, central_k, idx_l)
    return None
