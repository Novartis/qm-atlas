import copy
import logging
from collections.abc import Callable
from functools import partial
from itertools import combinations, permutations
from multiprocessing import Pool
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import TransformConformer
from rdkit.Numerics import rdAlignment

from qm_atlas.utils import remove_conformers

_logger = logging.getLogger(__name__)

_PERIODIC_TABLE = Chem.GetPeriodicTable()

# Criteria used by check_geometry_consistency. The length limits are expressed as
# multiples of the sum of the two covalent radii, which are taken from rdkit, so
# that no per-element-pair bond lengths have to be maintained here. The values
# were calibrated on 803 optimized conformers (b-p/def2-TZVP), for which the
# longest intact bond reaches 1.10 and the closest non-bonded contact 1.38 times
# the sum of the covalent radii.
BOND_STRETCH_SCALE = 1.35
NONBONDED_CONTACT_SCALE = 1.15

GeometryViolationKind = Literal["stretched_bond", "unexpected_contact"]


def join_molecules(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    prb_cids: list[int] | None = None,
    ref_cids: list[int] | None = None,
    reset_idcs: bool = True,
) -> tuple[Chem.Mol, dict[int, int]]:
    """Joins two rdkit molecules representing the same structure, which can have
        different indexings and several conformations each.

    Args:
        prb_mol (Chem.Mol):
            The molecule to add to a reference (atom indices of the reference are kept)
        ref_mol (Chem.Mol):
            The reference molecule.
        prb_cids (list[int], optional):
            The conformer IDs of the probe molecule to keep. Default to None, in this
            case all are kept.
        ref_cids (list[int], optional):
            The conformer IDs of the reference molecule to keep. Default to None, in this
            case all are kept.
        reset_idcs (bool, optional):
            If True, re-assigns conformer IDs sequentially in the joined molecule.
            Defaults to True.

    Returns:
        (Chem.Mol):
            The joined molecule with the atom ordering of the reference and all conformers
        (dict[int, int]):
            A dictionary mapping previous IDs in the probe molecule to the new ids in
            the the joined molecule.
    """
    if prb_cids is None:
        prb_cids = [conf.GetId() for conf in prb_mol.GetConformers()]

    mol_int = copy.deepcopy(ref_mol)

    all_ref_cids = [conf.GetId() for conf in ref_mol.GetConformers()]
    if ref_cids is None:
        ref_cids = all_ref_cids
    else:
        ref_cids_set = set(ref_cids)
        idcs_to_remove = set([idx for idx in all_ref_cids if idx not in ref_cids_set])
        remove_conformers(mol_int, idcs_to_remove, reset_idcs=reset_idcs)

    # add the second molecule as a conformer
    cid_map = dict()
    atom_map = ref_mol.GetSubstructMatch(prb_mol)
    for prb_cid in prb_cids:
        prb_conformer = prb_mol.GetConformer(prb_cid)
        new_conformer = Chem.Conformer(mol_int.GetConformer(0))
        for prb_idx, ref_idx in enumerate(atom_map):
            position = prb_conformer.GetAtomPosition(prb_idx)
            new_conformer.SetAtomPosition(ref_idx, position)
        new_id = mol_int.AddConformer(new_conformer, assignId=True)
        cid_map[prb_cid] = new_id

    return mol_int, cid_map


def align_conformers(mol: Chem.Mol, ref_id: int | None = None) -> dict[int, float]:
    """Aligns all conformers in a molecule to a reference conformation.

    Args:
        mol (Chem.Mol): The molecule.
        ref_id (int, optional):
            The conformer ID of the reference conformation. Defaults to None, in this
            case the first conformer is used as a reference.

    Returns:
        rmsd_dict (dict[int, float]):
            A dictionary mapping the conformer ID to the RMSD to the reference conformation.
    """
    num_atoms = mol.GetNumAtoms()
    h_conf_rmsd = HydrogenConformerRMSD(mol)
    conf_ids = [conformer.GetId() for conformer in mol.GetConformers()]

    if ref_id is None:
        ref_id = conf_ids[0]

    rmsd_dict = dict()
    for prb_id in conf_ids:

        if prb_id == ref_id:
            rmsd_dict[prb_id] = 0
            continue

        rmsd, transf, full_match = h_conf_rmsd.get_conformer_rmsd((ref_id, prb_id))
        rmsd_dict[prb_id] = rmsd
        atom_positions = [
            mol.GetConformer(prb_id).GetAtomPosition(atom_idx) for atom_idx in range(num_atoms)
        ]
        conformer = mol.GetConformer(prb_id)
        for ref_idx, prb_idx in enumerate(full_match):
            if prb_idx is None:
                raise RuntimeError("Indexing Issue occurred in conformer alignment.")
            prb_position = atom_positions[prb_idx]
            conformer.SetAtomPosition(ref_idx, prb_position)
        TransformConformer(conformer, transf)

    return rmsd_dict


def align_mol_to_reference(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    prb_cids: list[int] | None = None,
    ref_cid: int = 0,
) -> tuple[Chem.Mol, dict]:
    """Aligns a Molecule to a reference and calculates the RMSD. This code separates
    the hydrogens for the calculation of the RMSD, this is an (often very good)
    approximation, which avoids the combinatorial explosion occuring when applying
    GetBestRMS to molecules with hydrogens.

    Args:
        prb_mol (Chem.Mol):
            The molecule to align to the reference
        ref_mol (Chem.Mol):
            The reference molecule
        prb_cids (list[int], optional):
            The conformer IDs in the probe molecule to align. Defaults to None
            (all conformers).
        ref_cid (int, optional):
            The index of the conformer in the reference molecule to consider. Defaults to 0.

    Returns:
        aligned_mol (Chem.Mol):
            A molecule with all requested conformations of the probe molecule
            (keeping their original conformer IDs), aligned to the reference conformation.
        rmsd_dict (Dict[int, float]):
            A dictionary mapping the conformer IDs to the RMSD to the reference conformation.
    """

    # we create a new molecule that has all conformations of the probe molecule
    # and the reference conformation added to it

    if prb_cids is None:
        prb_cids = [conf.GetId() for conf in prb_mol.GetConformers()]

    mol_int, cid_map = join_molecules(ref_mol, prb_mol, ref_cids=prb_cids, prb_cids=[ref_cid])
    # the probe conformers have kept their conformer IDS, the conformer ID of the
    # reference conformation can be read off from the cid_map.
    rmsd_dict = align_conformers(mol_int, ref_id=cid_map[ref_cid])

    mol_int.RemoveConformer(cid_map[ref_cid])
    del rmsd_dict[cid_map[ref_cid]]

    return mol_int, rmsd_dict


class ConformerRMSD:
    """This is an adapted version of rdkit.Chem.rdMolAlign.GetBestAlignmentTransform,
    made to be applied to conformers of the same molecule, while not repeating the
    substructure matching needed to enumerate the symmetries of the molecule.
    """

    UNIQUIFY = False
    RECURSION_POSSIBLE = True
    USE_CHIRALITY = False
    USE_QUERY_QUERY_MATCHES = False

    def __init__(
        self,
        mol: Chem.Mol,
        max_matches: int = 1e6,
        weights: list | None = None,
        reflect: bool = False,
        max_iters: int = 50,
    ):

        if weights is None:
            weights = []

        self.mol = mol
        self.max_matches = max_matches
        self.weights = weights
        self.reflect = reflect
        self.max_iters = max_iters
        self.max_matches = max_matches
        self.matches = self.find_matches()

    def find_matches(self) -> list[tuple[int]]:
        """finds all symmetries of the input molecule

        Returns:
            matches (Tuple[Tuple[int]]):
                for each tuple, mapping idx -> match[idx] gives a
                symmetry of the molecule.
        """
        sub_match_ps = Chem.SubstructMatchParameters()
        sub_match_ps.uniquify = self.UNIQUIFY
        sub_match_ps.recursionPossible = self.RECURSION_POSSIBLE
        sub_match_ps.useChirality = self.USE_CHIRALITY
        sub_match_ps.useQueryQueryMatches = self.USE_QUERY_QUERY_MATCHES
        sub_match_ps.maxMatches = self.max_matches
        matches = self.mol.GetSubstructMatches(self.mol, sub_match_ps)
        if len(matches) > self.max_matches:
            name = self.mol.GetProp("_Name") if self.mol.HasProp("_Name") else "UNNAMED"
            _logger.warning("Warning in Calculation of Conformer RMSD:")
            _logger.warning(f"{len(matches)} matches detected for molecule {name}.")
        matches = [match for match in matches if self._match_is_valid(match)]
        return matches

    def _match_is_valid(self, match):
        for idx, matched_idx in enumerate(match):
            atom_1, atom_2 = self.mol.GetAtomWithIdx(idx), self.mol.GetAtomWithIdx(matched_idx)
            if not atom_1.GetTotalNumHs() == atom_2.GetTotalNumHs():
                return False
        return True

    def align_confs_on_atom_map(self, match: tuple[int], conf_idx_1: int, conf_idx_2: int):
        """finds the minimal MSD for the atoms in two conformers, given a matching.

        Args:
            match (Tuple[int]):
                A mapping of atoms in the first conformer to atoms in the second conformer.
            conf_idx_1 (int): index of the first conformer
            conf_idx_2 (int): index of the second conformer

        Returns:
            msd (float):
                The minimal mean squared distance between matched points, given an
                optimal alignment.
            trans (np.array):
                A 4x4 matrix specifying the transformation to be applied
                to the second conformer to align it to the first.
        """
        conf_1 = self.mol.GetConformer(conf_idx_1)
        conf_2 = self.mol.GetConformer(conf_idx_2)
        points_1 = [conf_1.GetAtomPosition(i) for i in range(len(match))]
        points_2 = [conf_2.GetAtomPosition(i) for i in match]
        ssr, trans = rdAlignment.GetAlignmentTransform(
            points_1, points_2, self.weights, self.reflect, self.max_iters
        )
        return ssr / len(points_1), trans

    def get_best_rmsd(self, conf_idx_1: int, conf_idx_2: int):
        """finds the minimal rmsd of two conformers, taking symmetries into account.

        Args:
            conf_idx_1 (int): index of the first conformer
            conf_idx_2 (int): index of the first conformer

        Returns:
            min_rmsd (float):
                The minimal RMSD between the two conformers,
            best_trans (np.array):
                The transformation to apply to the second confomer to get an
                optimal alignment
            best_match (Tuple[int]):
                The matching of atoms in the conformers that leads to an optimal
                alignment, best_match[idx] gives the atom in the second conformer to
                be matched to the corresponding one in the fist conformer.
        """
        if not self.matches:
            raise RuntimeError("No atom matches available for alignment")
        best_msd = np.inf
        best_match = None
        best_trans = None
        for match in self.matches:
            msd, tmp_trans = self.align_confs_on_atom_map(match, conf_idx_1, conf_idx_2)
            if msd < best_msd:
                best_msd = msd
                best_match = match
                best_trans = tmp_trans
        if best_match is None:
            raise RuntimeError("Failed to find a valid atom match for alignment")
        if best_trans is None:
            raise RuntimeError("Failed to compute a valid transformation for alignment")
        return np.sqrt(best_msd), best_trans, best_match


class HydrogenConformerRMSD:
    """This class allows to consider symmetries of molecules for the calculation
    of RMSDs between conformers, without running into the combinatorial explosion
    typically observed for molecules with hydrogens.

    It first finds an optimal match for the rmsd between two conformations. Based
    on this alignment, it tries to figure out how to best match the hydrogen atoms,
    considering the hydrogens attached to matched heavy atoms separately.
    This optimal match is then used to calculate the final RMSD.

    Note that this should be an (often exact) approximation to the RMSD value
    obtained by rdkit.Chem.rdMolAlign.GetBestRMS, but there can be cases where
    the matching found in this way is not optimal.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        max_matches: int = 1000000,
        weights: list | None = None,
        reflect: bool = False,
        max_iters: int = 50,
    ):

        if weights is None:
            weights = []

        # molecules
        self.mol = Chem.Mol(mol)
        self._mol_no_h = Chem.RemoveAllHs(mol)

        # use any match to related indices in the molecules with and without hydrogens
        self._match_h_removed = self.mol.GetSubstructMatch(self._mol_no_h)

        # indices of conformer pairs, lazy initalization
        self._combinations = list()
        # self._list_idx_to_conf_idcs = list()
        self._conf_idcs_to_list_idx = dict()

        # settings for RMSD calculations and substructure matching
        self.max_matches = max_matches
        self.weights = weights
        self.reflect = reflect
        self.max_iters = max_iters

        # instance to find optimal matches of heavy atoms
        self.conf_rmsd = ConformerRMSD(
            self._mol_no_h,
            max_matches=self.max_matches,
            weights=self.weights,
            reflect=self.reflect,
            max_iters=self.max_iters,
        )

        # getting hydrogen indices from heavy atom indices
        self.hydrogen_map = self._get_hydrogen_map()

    def _get_hydrogen_map(self) -> list[list[int]]:
        """Pre-computes the map of each heavy atom to the hydrogens which are bonded to it.

        Returns:
            hydrogen_map(List[List[int]]):
                hydrogen_map[heavy_idx] is the list of indices of the hydrogen atoms
                connected to the heavy atom with index heavy_idx (in the full molecule).
        """

        hydrogen_map = list()

        for atom in self.mol.GetAtoms():
            hydrogen_idcs = [at.GetIdx() for at in atom.GetNeighbors() if at.GetSymbol() == "H"]
            hydrogen_map.append(hydrogen_idcs)

        return hydrogen_map

    def _extend_match(
        self, heavy_match: list[int], transf: np.ndarray, conf_idx_1: int, conf_idx_2: int
    ) -> list[int | None]:
        """Extends a matching of heavy atoms to a full matching of all atoms.

        Args:
            heavy_match (List[int]:
                Mapping heavy atom indices in the first conformer to the corresponding
                ones in the second conformer
            transf (np.array):
                The transformation matrix that optimally aligns the heavy atoms in the
                second conformer to the ones in the first conformer.
            conf_idx_1 (int):
                The index of the first conformer
            conf_idx_2 (int):
                The index of the second conformer

        Returns:
            full_match (List[int]): The full mapping of atom indices.
        """

        full_match: list[int | None] = [None for _ in range(self.mol.GetNumAtoms())]
        for no_h_idx_1, no_h_idx_2 in enumerate(heavy_match):
            idx_1 = self._match_h_removed[no_h_idx_1]
            idx_2 = self._match_h_removed[no_h_idx_2]
            full_match[idx_1] = idx_2

        conf_1 = self.mol.GetConformer(conf_idx_1)
        conf_2 = Chem.Conformer(self.mol.GetConformer(conf_idx_2))
        TransformConformer(conf_2, transf)

        coord_1: np.ndarray = conf_1.GetPositions()
        coord_2: np.ndarray = conf_2.GetPositions()

        for heavy_idx_1 in self._match_h_removed:
            heavy_idx_2 = full_match[heavy_idx_1]
            h_idcs_1 = self.hydrogen_map[heavy_idx_1]
            h_idcs_2 = self.hydrogen_map[heavy_idx_2]

            if len(h_idcs_1) != len(h_idcs_2):
                raise RuntimeError(f"Illegal Match Found: {conf_idx_1}, {conf_idx_2}")

            if len(h_idcs_1) == 0:
                continue
            if len(h_idcs_2) == 1:
                full_match[h_idcs_1[0]] = h_idcs_2[0]
                continue

            h_coords_1 = coord_1[h_idcs_1, :]
            min_permutation = None
            min_msd = np.inf

            for permuted_idcs in permutations(h_idcs_2):
                h_coords_2 = coord_2[permuted_idcs, :]
                msd = np.sum(np.square(h_coords_1 - h_coords_2))
                if msd < min_msd:
                    min_msd = msd
                    min_permutation = permuted_idcs

            for idx_1, idx_2 in zip(h_idcs_1, min_permutation):
                full_match[idx_1] = idx_2

        return full_match

    def get_combinations(self):
        if self._combinations:
            return self._combinations

        conf_idcs = sorted([conf.GetId() for conf in self.mol.GetConformers()])

        for list_idx, (idx_1, idx_2) in enumerate(combinations(conf_idcs, 2)):
            self._combinations.append((idx_1, idx_2))
            # self._list_idx_to_conf_idcs.append((idx_1, idx_2))
            self._conf_idcs_to_list_idx[(idx_1, idx_2)] = list_idx

        return self._combinations

    def get_conf_idcs(self, list_idx: int) -> tuple[int, int]:
        """Maps an index for the list returned by get_rmsd_matrix to the pair of
        conformer indices.
        """
        if not self._combinations:
            self.get_combinations()

        return self._combinations[list_idx]

    def get_list_idx(self, conf_idcs: tuple[int, int]) -> int:
        """Maps a pair of conformer indices to the respective list index for the list
        returned by get_rmsd_matrix
        """
        if not self._combinations:
            self.get_combinations()

        return self._conf_idcs_to_list_idx[conf_idcs]

    def get_rmsd_matrix(self, num_cores: int = 1) -> list[float]:
        """Calculates the matrix of rmsd values between all conformations of the molecule.

        Args:
            num_cores (int, optional):
                The number of cores to use. Defaults to 1.

        Returns:
            rmsd_matrix (List[int]):
                A list holding all rmsd values between conformers of the molecule.
                See get_conf_idcs() and get_list_idx() for information on the indexing.
        """
        # Note: Conformers might not be labelled consecutively.
        if not self._combinations:
            self.get_combinations()

        if num_cores == 1:
            rmsd_matrix = [self._get_conf_rmsd(idcs) for idcs in self._combinations]

        else:
            with Pool(num_cores) as pool:
                rmsd_matrix = pool.map(self._get_conf_rmsd, self._combinations)

        return rmsd_matrix

    def _get_conf_rmsd(self, conf_idcs) -> float:
        return self.get_conformer_rmsd(conf_idcs)[0]

    def get_conformer_rmsd(
        self, conf_idcs: tuple[int, int]
    ) -> tuple[float, np.ndarray, list[int | None]]:
        """Calculates the RMSD between two conformations.

        Args:
            conf_idcs (Tuple[int, int]): _description_

        Returns:
            rmsd (float): The RMSD value
            transf (np.array): The transformation that aligns the second conformer to the first.
            match (list[int | None]): The optimal match of atoms in the two conformers.
        """

        conf_1 = self.mol.GetConformer(conf_idcs[0])
        conf_2 = self.mol.GetConformer(conf_idcs[1])
        # find best heavy atom match
        _, transf, match = self.conf_rmsd.get_best_rmsd(*conf_idcs)

        full_match = self._extend_match(match, transf, *conf_idcs)

        # calculate final rmsd
        points_1 = [conf_1.GetAtomPosition(idx_1) for idx_1 in range(len(full_match))]
        points_2 = [conf_2.GetAtomPosition(idx_2) for idx_2 in full_match]
        ssr, transf = rdAlignment.GetAlignmentTransform(
            points_1, points_2, self.weights, self.reflect, self.max_iters
        )

        return np.sqrt(ssr / len(full_match)), transf, full_match


def mat_to_list_idx(mat_idcs: tuple[int, int]) -> int | None:
    """translates the indices accessing a lower_triangular matrix to the index of the
    corresponding list in the way that it is assumed e.g. in
    rdkit.Chem.AllChem.GetConformerRMSMatrix

    The list indices correspond as follows:
    [(0,1), (0,2), (1,2), (0,3), (1,3), (2,3), ... ]

    Args:
        mat_idcs (tuple[int, int]): The (i, j) matrix indices.

    Returns:
        int | None: The corresponding list index, or None if i == j.
    """
    i, j = mat_idcs

    if i == j:
        return None
    if i > j:
        i, j = j, i

    list_idx = j * (j - 1) // 2 + i
    return list_idx


def list_to_mat_idx(list_idx: int) -> tuple[int, int]:
    """translates a list index to the corresponding two indices for a lower triangular
    matrix. See mat_to_list_idx for details

    Args:
        list_idx (int): The list index

    Returns:
        i, j (Tuple[int, int]): The two matrix indices
    """
    j = int(1 / 2 + np.sqrt(2 * list_idx + 1 / 4))
    i = list_idx - int(j * (j - 1) / 2)
    return i, j


def get_rmsd_matrix(
    mol: Chem.Mol,
    method: str = "tradeoff",
    num_cores: int = 1,
) -> tuple[list[float], Callable[[int], tuple[int, int]]]:
    """Calculates the RMSD matrix for all conformer pairs in a molecule.

    Args:
        mol (Chem.Mol):
            The rdkit molecule (should have conformers set)
        method (str):
            The method to use to calculate the RMSD. Options are "exact", "tradeoff",
            "fast", and "strip-hydrogens". Default is "tradeoff".
        num_cores (int, optional):
            The number of cores to use. Defaults to 1.

    Returns:
        rmsd_matrix (List[float]):
            A list of RMSD values for all conformer pairs.
        idx_map (Callable[[int], Tuple[int, int]]):
            A callable that maps a list index to a pair of conformer indices.
    """
    _mol = Chem.Mol(mol)

    if method == "exact":
        h_conf_rmsd = HydrogenConformerRMSD(_mol)
        _combinations = h_conf_rmsd.get_combinations()
        idx_map = h_conf_rmsd.get_conf_idcs

        get_rmsd_local = partial(get_rdkit_best_rmsd, mol=_mol)

        if num_cores == 1:
            rmsd_matrix = [get_rmsd_local(idcs) for idcs in _combinations]
        else:
            with Pool(num_cores) as pool:
                rmsd_matrix = pool.map(get_rmsd_local, _combinations)

    elif method == "strip-hydrogens":
        mol_no_h = Chem.RemoveHs(_mol)
        conf_ids = sorted([conf.GetId() for conf in mol_no_h.GetConformers()])
        _combinations = list(combinations(conf_ids, 2))
        idx_map = _combinations.__getitem__

        get_rmsd_local = partial(get_rdkit_best_rmsd, mol=mol_no_h)

        if num_cores == 1:
            rmsd_matrix = [get_rmsd_local(idcs) for idcs in _combinations]
        else:
            with Pool(num_cores) as pool:
                rmsd_matrix = pool.map(get_rmsd_local, _combinations)

    elif method == "tradeoff":
        try:
            h_conf_rmsd = HydrogenConformerRMSD(_mol)
            rmsd_matrix = h_conf_rmsd.get_rmsd_matrix(num_cores=num_cores)
            idx_map = h_conf_rmsd.get_conf_idcs
        except RuntimeError:
            _logger.error("Tradeoff RMSD calculation failed. Falling back to fast method")
            return get_rmsd_matrix(mol, method="fast", num_cores=num_cores)

    elif method == "fast":
        rmsd_matrix = AllChem.GetConformerRMSMatrix(_mol)
        idx_map = list_to_mat_idx

    else:
        raise ValueError(f"Unknown method {method}")

    return rmsd_matrix, idx_map


def get_duplicate_idcs(
    mol: Chem.Mol,
    rms_threshold: float,
    method: str = "tradeoff",
    num_cores: int = 1,
) -> set[int]:
    """Finds all indices of conformers in a molecule, which can be considered duplicates.
        These are all conformers which have an RMSD below threshold to any previous
        conformer, which is not considered a duplicate itself.

    Args:
        mol (Chem.Mol):
            The rdkit molecule (should have conformers set)
        rms_threshold (float):
            The threshold value for the RMSD.
        method (str):
            The method to use to calculate the RMSD. Options are "exact", "tradeoff",
            "fast", and "strip-hydrogens". Default is "tradeoff".
        num_cores (int, optional):
            The number of cores to use. Defaults to 1.

    Returns:
        set[int]: The set of conformer indices which are duplicates
    """
    rmsd_matrix, idx_map = get_rmsd_matrix(mol, method=method, num_cores=num_cores)

    confs_to_remove = set()
    for list_idx, rmsd_val in enumerate(rmsd_matrix):
        if rmsd_val < rms_threshold:
            previous_conf_idx, current_conf_idx = idx_map(list_idx)
            if previous_conf_idx not in confs_to_remove:
                confs_to_remove.add(current_conf_idx)
    return confs_to_remove


def get_rdkit_best_rmsd(idcs, mol):
    """Return the best-RMS overlay between two conformers of *mol* (RDKit GetBestRMS)."""
    return Chem.rdMolAlign.GetBestRMS(
        mol,
        mol,
        refId=idcs[0],
        prbId=idcs[1],
        symmetrizeConjugatedTerminalGroups=False,
    )


def deduplicate_conformers(
    mol: Chem.Mol,
    rms_threshold: float,
    method: str = "tradeoff",
    num_cores: int = 1,
) -> None:
    """Finds all duplicate conformers and removes them. The molecule is modified in
        place.

    Args:
        mol (Chem.Mol):
            The rdkit molecule (should have conformers set)
        rms_threshold (float):
            The threshold value for the RMSD below which two conformers are
            considered duplicates.
        method (str):
            The method to use to calculate the RMSD. Options are "exact", "tradeoff",
            "fast", and "strip-hydrogens". Default is "tradeoff".
        num_cores (int):
            Number of CPU cores to use for the pairwise RMSD calculation. Defaults to 1.
    """

    confs_to_remove = get_duplicate_idcs(
        mol, rms_threshold=rms_threshold, method=method, num_cores=num_cores
    )
    remove_conformers(mol, confs_to_remove)


def covalent_distance_limits(mol: Chem.Mol, scale: float) -> np.ndarray:
    """Builds a matrix of pairwise distance limits from the element covalent radii.

    The limit for a pair of atoms is ``scale * (r_cov_i + r_cov_j)``, with the
    covalent radii taken from rdkit's periodic table. This avoids maintaining a
    table of per-element-pair reference bond lengths in this module.

    Args:
        mol (Chem.Mol): The molecule whose elements determine the radii.
        scale (float): The factor applied to the sum of the two covalent radii.

    Returns:
        np.ndarray: An array of shape (num_atoms, num_atoms) of distance limits.
    """
    radii = np.array(
        [_PERIODIC_TABLE.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    )
    return scale * (radii[:, None] + radii[None, :])


def check_geometry_consistency(
    mol: Chem.Mol,
    conf_id: int = -1,
    bond_stretch_scale: float = BOND_STRETCH_SCALE,
    contact_scale: float | None = NONBONDED_CONTACT_SCALE,
) -> list[GeometryViolationKind]:
    """Checks whether a set of coordinates is still consistent with the bond table.

    The bond table of the molecule is taken as the truth - connectivity is never
    re-perceived from the coordinates. Two criteria are applied, both of them
    relative to the sum of the covalent radii of the two elements involved
    (see :func:`covalent_distance_limits`):

    1. ``stretched_bond``: every declared bond must be shorter than
       ``bond_stretch_scale * (r_cov_i + r_cov_j)``, so a bond that was broken
       during the optimization is detected.
    2. ``unexpected_contact``: no pair of atoms that is *not* bonded in the bond
       table may come closer than ``contact_scale * (r_cov_i + r_cov_j)``, so a
       bond that was formed during the optimization is detected. This includes
       1-3 pairs, since ring closures onto a neighboring atom are a common outcome.

    Args:
        mol (Chem.Mol): The rdkit mol to check. Its bond table defines the
            expected connectivity, hydrogens have to be explicit.
        conf_id (int, optional): The id of the conformer to check. Defaults to -1,
            which uses the only/first conformer.
        bond_stretch_scale (float, optional): Scale factor for criterion 1.
            Defaults to :data:`BOND_STRETCH_SCALE`.
        contact_scale (float | None, optional): Scale factor for criterion 2,
            pass None to skip the test. Defaults to :data:`NONBONDED_CONTACT_SCALE`.

    Returns:
        list[GeometryViolationKind]: The kinds of inconsistency which were
            detected, empty if the coordinates are consistent with the bond table.
    """
    num_atoms = mol.GetNumAtoms()
    dist_mat = np.asarray(Chem.rdmolops.Get3DDistanceMatrix(mol, confId=conf_id))

    bonded = np.zeros((num_atoms, num_atoms), dtype=bool)
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[begin_idx, end_idx] = bonded[end_idx, begin_idx] = True

    violations: list[GeometryViolationKind] = []

    # 1. declared bonds must not be stretched beyond a covalent bond length
    stretch_limits = covalent_distance_limits(mol, bond_stretch_scale)
    if np.any(bonded & (dist_mat > stretch_limits)):
        violations.append("stretched_bond")

    # 2. atoms which are not bonded must not be within covalent bonding distance
    if contact_scale is not None:
        contact_limits = covalent_distance_limits(mol, contact_scale)
        # numpy upper triangular (np.triu) removes the diagonal
        if np.any(np.triu(~bonded & (dist_mat < contact_limits), 1)):
            violations.append("unexpected_contact")

    return violations


def sanity_check(
    mol: Chem.Mol,
    conf_id: int,
) -> bool:
    """Checks whether a conformer is still consistent with the bond table of the
    molecule, i.e. whether all declared bonds are intact and no new bond has been
    formed. See :func:`check_geometry_consistency` for the criteria which are
    applied.

    Args:
        mol (Chem.Mol): The rdkit mol to check
        conf_id (int): The id of the conformer to check

    Returns:
        bool: whether the check was passed
    """
    return not check_geometry_consistency(mol, conf_id)
