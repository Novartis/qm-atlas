import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from ppqm import chembridge
from rdkit import Chem

from qm_atlas import utils

_logger = logging.getLogger(__name__)


@dataclass
class PkaCenter:
    center_type: str
    index: int
    pka_value: float
    parent_name: str | None = None
    child_name: str | None = None

    def __post_init__(self):
        if self.center_type not in ("a", "b"):
            raise ValueError(f"Illegal center type {self.center_type} encountered")

    def __lt__(self, other) -> bool:
        """compares two pKa centers based on their pKa values, used for sorting lists of pKa centers.
        Args:
            other (PkaCenter): The other pKa center to compare to
        Returns:
            bool: True if this pKa center has a lower pKa value than the other
        """
        return self.pka_value < other.pka_value

    def is_relevant(self, lower_ph: float, upper_ph: float) -> bool:
        """checks if the affects the ionization states to be considered for a given pH range.
        If the pka_value is nan, it is considered to be relevant.
        Args:
            lower_ph (float): The lower pH value
            upper_ph (float): The upper pH value
        Returns:
            bool: True if the pKa center is in the given pH range
        """
        if self.center_type == "a":
            return not (self.pka_value > upper_ph)
        else:
            return not (self.pka_value < lower_ph)

    def is_in_ph_range(self, lower_ph: float, upper_ph: float) -> int:
        """checks if the pKa center is in the given pH range. If the pka value is nan,
        it is considered to be in the range.
        Args:
            lower_ph (float): The lower pH value
            upper_ph (float): The upper pH value
        Returns:
            bool: True if the pKa center is in the given pH range
        """
        is_low_enough = not (self.pka_value > upper_ph)
        is_high_enough = not (self.pka_value < lower_ph)
        return is_low_enough and is_high_enough


def get_pka_centers_from_arrays(
    center_types: Iterable[str],
    centers: Iterable[int],
    pkas: Iterable[float],
) -> list[PkaCenter]:
    """
    Get a list of PkaCenter objects from the given arrays. The arrays are expected to be of the same length.

    Args:
        center_types (np.ndarray):
            The types of the centers ("a" for acid, "b" for base)
        centers (np.ndarray):
            The atom indices of the centers
        pkas (np.ndarray):
            The pKa values of the centers

    Returns:
        list[PkaCenter]:
            A list of PkaCenter objects
    """
    pka_centers = [
        PkaCenter(str(center_type), int(center_idx), float(pka))
        for center_type, center_idx, pka in zip(center_types, centers, pkas)
    ]
    return pka_centers


class PkaCalculationInfo:
    def __init__(self, compound_name: str):
        self.compound_name = compound_name

        # lazy initialization
        self.name_to_mol: dict[str, tuple[Chem.Mol, bool]] = dict()
        self.pka_centers: dict[int, PkaCenter] = dict()

        # helper dictionaries for efficient lookup
        self._name_to_smiles: dict[str, str] = dict()
        self._smiles_to_name: dict[str, str] = dict()
        self._suffix_count: defaultdict[str, int] = defaultdict(lambda: 0)

    def is_contained(self, mol: Chem.Mol) -> bool:
        """checks if the molecule is contained in the dictionary of compounds"""
        smi = Chem.MolToSmiles(mol)
        return smi in self._name_to_smiles.values()

    def _assign_name(self, mol: Chem.Mol) -> str:
        """assigns a name to the molecule based on its charge, follows the rescoss conventions"""
        rescoss_suffix = get_charge_suffix(mol)
        self._suffix_count[rescoss_suffix] += 1
        count = self._suffix_count[rescoss_suffix]
        if count != 1:
            rescoss_suffix += str(count)

        mol_name = f"{self.compound_name}{rescoss_suffix}"
        return mol_name

    def add_compound(
        self,
        mol: Chem.Mol,
        require_conformer_expansion: bool = True,
    ) -> str:
        """adds a compound to the dictionary of compounds"""
        smi = Chem.MolToSmiles(mol)
        if smi not in self._smiles_to_name:
            mol_name = self._assign_name(mol)
            mol.SetProp("_Name", mol_name)
            self.name_to_mol[mol_name] = (mol, require_conformer_expansion)
            self._name_to_smiles[mol_name] = smi
            self._smiles_to_name[smi] = mol_name
            return mol_name
        else:
            # update the require_conformer_expansion flag
            name = self._smiles_to_name[smi]
            mol.SetProp("_Name", name)
            self.name_to_mol[name] = (mol, require_conformer_expansion)

        return self._smiles_to_name[smi]

    def add_pka_center(
        self,
        pka_center: PkaCenter,
        parent_compound: Chem.Mol,
        child_compound: Chem.Mol,
        require_conformer_expansion: bool = True,
    ) -> None:

        """adds a pKa center to the class"""

        if pka_center.index in self.pka_centers:
            raise ValueError(f"pKa center {pka_center.index} already exists")

        parent_name = self.add_compound(
            parent_compound, require_conformer_expansion=require_conformer_expansion
        )
        child_name = self.add_compound(
            child_compound, require_conformer_expansion=require_conformer_expansion
        )
        pka_center.parent_name = parent_name
        pka_center.child_name = child_name

        self.pka_centers[pka_center.index] = pka_center


def generate_ionic_species(
    mol: Chem.Mol,
    pka_centers: list[PkaCenter],
    lower_ph_threshold: float = -np.inf,
    upper_ph_threshold: float = np.inf,
) -> PkaCalculationInfo:
    """
    Determine the ionic species of the molecule based on the provided pka centers. Proceeds in order of the
    pKa values, protonating the basic centers and deprotonating the acidic centers. The species to be generated can thus
    have multiple charges, each charge appearing only once.

    Args:
        mol (Chem.Mol):
            The molecule to be ionized
        pka_centers (list[PkaCenter]):
            a list of pKa centers, each with a type, and atom index, and a pKa value.
        lower_ph_threshold (float, optional):
            Only consider pKa centers relevant at or above this pH. Defaults to -inf.
        upper_ph_threshold (float, optional):
            Only consider pKa centers relevant at or below this pH. Defaults to inf.

    Returns:
        PkaCalculationInfo: A :class:`PkaCalculationInfo` object containing the
            information needed for downstream pKa calculations.
    """

    # filter out the pKa centers that are not in the given pH range
    pka_centers = [
        pka_center
        for pka_center in pka_centers
        if pka_center.is_relevant(lower_ph_threshold, upper_ph_threshold)
    ]

    # in order of pk value
    pka_centers_sorted = sorted(pka_centers)
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNNAMED"

    orig_mol = Chem.AddHs(mol)
    utils.store_atom_indices(orig_mol)
    edit_mol = Chem.RemoveHs(orig_mol)
    original_idcs_to_new_idcs = utils.get_atom_indices_map(edit_mol)

    # protonate all basic centers
    for pka_center in pka_centers_sorted:

        if pka_center.center_type == "a":
            continue

        # protonate the center
        original_idx = int(pka_center.index)
        new_idx = original_idcs_to_new_idcs[original_idx]
        atom = edit_mol.GetAtomWithIdx(new_idx)
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)

    edit_mol.UpdatePropertyCache()
    Chem.SanitizeMol(edit_mol)

    pka_calculation_info = PkaCalculationInfo(mol_name)
    excluded_species = list()

    previous_species = utils.restore_atom_indices(Chem.AddHs(edit_mol))
    previous_pka_value = -np.inf

    # deprotonate the ionizable centers one by one
    for pka_center in pka_centers_sorted:

        original_idx = int(pka_center.index)
        new_idx = original_idcs_to_new_idcs[original_idx]
        atom = edit_mol.GetAtomWithIdx(new_idx)

        # deprotonate the center
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        edit_mol.UpdatePropertyCache()
        Chem.SanitizeMol(edit_mol)
        new_species = utils.restore_atom_indices(Chem.AddHs(edit_mol))

        require_conformer_expansion = True
        if not pka_center.is_in_ph_range(lower_ph_threshold, upper_ph_threshold):
            require_conformer_expansion = False
            excluded_species.append(
                (
                    chembridge.copy_molobj(previous_species),
                    previous_pka_value,
                    pka_center.pka_value,
                )
            )

        if pka_center.center_type == "b":
            # de-protonated state is the parent for basic centers
            pka_calculation_info.add_pka_center(
                pka_center,
                new_species,
                previous_species,
                require_conformer_expansion=require_conformer_expansion,
            )
        else:
            # de-protonated state is the child
            pka_calculation_info.add_pka_center(
                pka_center,
                previous_species,
                new_species,
                require_conformer_expansion=require_conformer_expansion,
            )
        previous_species = new_species
        previous_pka_value = pka_center.pka_value

    excluded_species.append((chembridge.copy_molobj(previous_species), previous_pka_value, np.inf))

    # check if there are any excluded species, which are relevant in the given pH range,
    # but which have not been added so far, since they are not relevant for the pKa calculation
    for mol, lower_ph_value, higher_ph_value in excluded_species:
        # check if the excluded species is relevant in the given pH range
        if utils.intervals_overlap(
            (lower_ph_value, higher_ph_value),
            (lower_ph_threshold, upper_ph_threshold),
        ):
            pka_calculation_info.add_compound(mol, require_conformer_expansion=True)

    return pka_calculation_info


def generate_single_charge_ions(
    mol: Chem.Mol,
    pka_centers: list[PkaCenter],
    lower_ph_threshold: float = -np.inf,
    upper_ph_threshold: float = np.inf,
) -> PkaCalculationInfo:
    """
    Determine the ionic species of the molecule based on the provided pka centers. The function (de-)protonates each center
    separately, generating a singly charged ion for each center. The function does not take into account the pKa values of the centers.

    Args:
        mol (Chem.Mol):
            The molecule to be ionized
        pka_centers (list[PkaCenter]):
            a list of pKa centers, each with a type, and atom index, and a pKa value.
        lower_ph_threshold (float, optional):
            Only consider pKa centers relevant at or above this pH. Defaults to -inf.
        upper_ph_threshold (float, optional):
            Only consider pKa centers relevant at or below this pH. Defaults to inf.

    Returns:
        PkaCalculationInfo: A :class:`PkaCalculationInfo` object containing all the
            information needed for downstream pKa calculations.
    """
    pka_centers = [
        pka_center
        for pka_center in pka_centers
        if pka_center.is_in_ph_range(lower_ph_threshold, upper_ph_threshold)
    ]

    orig_mol = Chem.AddHs(mol)
    utils.store_atom_indices(orig_mol)
    edit_mol = Chem.RemoveHs(orig_mol)
    original_idcs_to_new_idcs = utils.get_atom_indices_map(edit_mol)

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNNAMED"
    pka_calculation_info = PkaCalculationInfo(mol_name)

    for pka_center in pka_centers:

        new_mol = chembridge.copy_molobj(edit_mol)
        original_idx = int(pka_center.index)
        new_idx = original_idcs_to_new_idcs[original_idx]
        atom = new_mol.GetAtomWithIdx(new_idx)

        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        if pka_center.center_type == "b":
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        elif pka_center.center_type == "a":
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        else:
            raise ValueError(f"Illegal center type {pka_center.center_type} encountered")
        new_mol.UpdatePropertyCache()
        Chem.SanitizeMol(new_mol)

        new_mol = utils.restore_atom_indices(Chem.AddHs(new_mol))
        pka_calculation_info.add_pka_center(
            pka_center,
            orig_mol,
            new_mol,
        )

    return pka_calculation_info


def get_charge_suffix(mol: Chem.Mol) -> str:
    """assigns rescoss suffix to a charge:
    _A for charge -1, _AA for -2, _BH for +1, _BHH for +2 and so on.
    """
    charge = Chem.rdmolops.GetFormalCharge(mol)
    if charge == 0:
        if utils.is_zwitterion(mol):
            return "_ZH"
        return ""
    elif charge < 0:
        return "_" + "A" * abs(charge)
    else:
        return "_B" + "H" * abs(charge)
