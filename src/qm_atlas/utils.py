import functools
import getpass
import json
import logging
import os
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ppqm import chembridge
from rdkit import Chem
from rdkit.Chem import SaltRemover

_logger = logging.getLogger(__name__)

open_utf8 = functools.partial(open, encoding="utf-8")


ATOM_IDX_PROP_KEY = "original_idx"
HYDROGEN_IDCS_PROP_KEY = "hydrogen_idcs"


def get_scratch_dir() -> Path:
    """Returns a scratch directory for temporary files. The function first checks
    if the environment variable TMPDIR is set and returns its value as a Path object.
    If TMPDIR is not set, it attempts to create a scratch directory in /scratch/<user>/qm_atlas_tmp,
    where <user> is the current user's username. If this fails due to permission issues, it falls back
    to creating a scratch directory in the current working directory.

    Raises:
        OSError:
            If the scratch directory cannot be created.

    Returns:
        Path: The path to the scratch directory.
    """
    if "TMPDIR" in os.environ:
        return Path(os.environ["TMPDIR"])

    user = getpass.getuser()
    scratch_dir = Path("/scratch") / user
    try:
        scratch_dir.mkdir(parents=False, exist_ok=True)
        mp_scratch_dir = scratch_dir / "qm_atlas_tmp"
        mp_scratch_dir.mkdir(parents=False, exist_ok=True)
        return mp_scratch_dir
    except (PermissionError, FileNotFoundError):
        _logger.warning(
            f"Could not create scratch directory in {scratch_dir}. "
            "Using current working directory instead."
        )
    try:
        scratch_dir = Path.cwd()
        mp_scratch_dir = scratch_dir / "qm_atlas_tmp"
        mp_scratch_dir.mkdir(parents=False, exist_ok=True)
        return mp_scratch_dir
    except OSError as exc:
        _logger.error(
            f"Permission denied to create local scratch directory in {scratch_dir}. "
            "Using current working directory instead."
        )
        raise exc


def are_equal(mol_1: Chem.Mol, mol_2: Chem.Mol) -> bool:
    """checks if two molecules are the same

    Args:
        mol_1 (Chem.Mol): the first molecule
        mol_2 (Chem.Mol): the seond molecule

    Returns:
        bool: whether they correspond to the same structure
    """

    smi_1 = Chem.MolToSmiles(Chem.RemoveHs(mol_1))
    smi_2 = Chem.MolToSmiles(Chem.RemoveHs(mol_2))

    return smi_1 == smi_2


def has_macrocycle(mol: Chem.Mol, ring_threshold: int = 10) -> bool:
    """Checks whether a molecule has a macrocycle, i.e. if the largest ring has more
        atoms than a given threshold

    Args:
        mol (Chem.Mol): The molecule to check
        ring_threshold (int): The threshold for the ring size. Defaults to 10.

    Returns:
        bool: whether the molecule contains a macrocycle

    """

    ri = mol.GetRingInfo()
    largest_ring_size = max((len(r) for r in ri.AtomRings()), default=0)

    return largest_ring_size >= ring_threshold


def conformer_is_3d(mol: Chem.Mol, conf_id: int = 0) -> bool:
    """Checks whether a molecule has 3D coordinates. The check is more extensive than simply checking
    for a non-zero z-coordinate. Instead it checks if there are any four consecutive atoms, for which
    the position vectors are linearly independent.

    Args:
        mol (Chem.Mol): The molecule to check

    Returns:
        bool: whether the molecule has 3D coordinates
    """
    if mol.GetNumConformers() == 0:
        return False

    atom_positions = mol.GetConformer(conf_id).GetPositions()

    ref_vec = atom_positions[0, :]
    rel_positions = atom_positions - ref_vec

    # three points lie in a plane going through the origin if they are linearly dependent
    for idx in range(4, rel_positions.shape[0] + 1):
        sub_arr = rel_positions[(idx - 3) : idx, :]
        determinant = np.linalg.det(sub_arr)
        if determinant > 1.0e-6:
            return True

    return False


def remove_conformers(
    mol: Chem.Mol,
    idcs_to_remove: Iterable[int],
    reset_idcs: bool = True,
) -> None:
    """Removes Conformers from an rdkit molecule object. Modifies the molecule in place.

    Args:
        mol (Chem.Mol): The molecule.
        idcs_to_remove (Iterable[int]): The conformer indices to be removed.
    """
    for idx in idcs_to_remove:
        mol.RemoveConformer(idx)

    if reset_idcs:
        for new_idx, conformer in enumerate(mol.GetConformers()):
            conformer.SetId(new_idx)


def has_hydrogens(
    mol: Chem.Mol,
):
    return mol.GetNumAtoms() == Chem.AddHs(mol).GetNumAtoms()


def as_float(value) -> float:
    """Return value as float, if possible. If the value cannot be parsed as a
    float, the original value is returned"""
    try:
        value = float(value)
    except (ValueError, TypeError):
        pass

    return value


def meta_func(func, args, **kwargs):
    """meta func to translate positional args to tuple"""
    return func(*args, **kwargs)


def store_atom_indices(mol: Chem.Mol):

    heavy_atom_idcs = dict()
    hydrogen_atom_idcs = defaultdict(list)
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        heavy_atom_idcs[idx] = atom.GetIdx()
        h_neighbors = [neighb for neighb in atom.GetNeighbors() if neighb.GetAtomicNum() == 1]
        for h_atom in h_neighbors:
            hydrogen_atom_idcs[idx].append(h_atom.GetIdx())

    for idx, heavy_atom_idx in heavy_atom_idcs.items():
        atom = mol.GetAtomWithIdx(idx)
        atom.SetIntProp(ATOM_IDX_PROP_KEY, heavy_atom_idx)
        atom.SetProp(HYDROGEN_IDCS_PROP_KEY, json.dumps(hydrogen_atom_idcs[idx]))


def get_atom_indices_map(mol: Chem.Mol) -> dict[int, int]:
    """returns a list mapping the original atom indices to the new ones.

    Args:
        mol (Chem.Mol): The molecule to get the atom indices from

    Returns:
        dict[int, int]: A dictionary mapping the original atom indices to the new ones.
    """

    original_idx_to_current_idx = dict()

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        new_idx = atom.GetIdx()
        old_idx = atom.GetIntProp(ATOM_IDX_PROP_KEY)
        original_idx_to_current_idx[old_idx] = new_idx

        hydrogen_idcs = json.loads(atom.GetProp(HYDROGEN_IDCS_PROP_KEY))
        h_neighbors = [neighb for neighb in atom.GetNeighbors() if neighb.GetAtomicNum() == 1]
        for h_atom, orig_idx in zip(h_neighbors, hydrogen_idcs):
            original_idx_to_current_idx[orig_idx] = h_atom.GetIdx()

    return original_idx_to_current_idx


def restore_atom_indices(mol: Chem.Mol) -> Chem.Mol:

    original_idx_to_current_idx = get_atom_indices_map(mol)

    new_mol = chembridge.copy_molobj(mol)
    for atom in new_mol.GetAtoms():
        atom.ClearProp(ATOM_IDX_PROP_KEY)
        atom.ClearProp(HYDROGEN_IDCS_PROP_KEY)

    # we have to give a new index for every atom, including the hydrogens which were added
    current_idx_to_new_idx = [-1] * new_mol.GetNumAtoms()
    for original_idx, current_idx in original_idx_to_current_idx.items():
        if original_idx < new_mol.GetNumAtoms():
            current_idx_to_new_idx[current_idx] = original_idx
    available_idcs = deque(set(range(new_mol.GetNumAtoms())) - set(current_idx_to_new_idx))

    for list_index in range(new_mol.GetNumAtoms()):
        if current_idx_to_new_idx[list_index] == -1:
            current_idx_to_new_idx[list_index] = available_idcs.popleft()

    new_idx_to_current_idx = [-1] * new_mol.GetNumAtoms()
    for current_idx, new_idx in enumerate(current_idx_to_new_idx):
        new_idx_to_current_idx[new_idx] = current_idx

    return Chem.RenumberAtoms(new_mol, new_idx_to_current_idx)


def is_zwitterion(mol: Chem.Mol) -> bool:
    """Check if a molecule is a zwitterion

    Args:
        mol (Chem.Mol): The molecule to check

    Returns:
        bool: True if the molecule is a zwitterion, False otherwise
    """
    return abs(Chem.GetFormalCharge(mol)) != sum(
        abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()
    )


def intervals_overlap(interval1: tuple[float, float], interval2: tuple[float, float]) -> bool:
    """Checks whether two intervals overlap. Touching counts as overlapping as well.
        For example, the intervals [1, 2] and [2, 3] overlap.

    Args:
        interval1 (tuple[float, float]):
            The first interval, values are expected to be in ascending order.
        interval2 (tuple[float, float]):
            The second interval, values are expected to be in ascending order.

    Returns:
        bool: True if the intervals overlap, False otherwise.
    """
    return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])


def remove_salt(smiles, saltfile: Path | None = None) -> str | None:
    """
    Remove salt and small molecules from smiles

    - Does not remove all non-covalented molecules
    - Does not neutralize the molecule

    >>> uncharger = rdMolStandardize.Uncharger()
    >>> mol = uncharger.uncharge(mol) # remove salt and adjust charges

    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    sRem = SaltRemover.SaltRemover(defnFilename=saltfile)
    mol = sRem.StripMol(mol)

    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol)

    # Remove duplicates
    smiles = smiles.split(".")
    smiles = set(smiles)
    smiles = ".".join(smiles)

    return smiles
