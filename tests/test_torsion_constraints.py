"""Tests for the heavy-atom selection of torsion constraints."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.wrappers import torsion_constraints


def build_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    """Embed and MMFF-relax a molecule so the geometry checks have something to work with."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def central_bonds(torsions) -> set[frozenset[int]]:
    return {frozenset(torsion[1:3]) for torsion in torsions}


def test_a_short_chain_yields_only_its_central_torsion():
    # n-butane: only the central C-C bond has a heavy neighbour on both sides.
    mol = build_mol("CCCC")
    torsions = torsion_constraints.find_heavy_atom_torsions(mol)
    assert len(torsions) == 1
    (torsion,) = torsions
    assert set(torsion) == {0, 1, 2, 3}
    # Consecutive atoms of a torsion must be bonded, and all four must be heavy.
    for first, second in zip(torsion, torsion[1:]):
        assert mol.GetBondBetweenAtoms(first, second) is not None
    assert all(mol.GetAtomWithIdx(idx).GetAtomicNum() > 1 for idx in torsion)


@pytest.mark.parametrize("smiles", ["C", "CC", "CCC", "CO", "CC(C)(C)C"])
def test_molecules_without_a_heavy_four_atom_chain_yield_no_torsions(smiles):
    # Each of these has no bond with a heavy neighbour on *both* sides.
    assert torsion_constraints.find_heavy_atom_torsions(build_mol(smiles)) == []


def test_hydrogen_rotors_are_not_central_bonds():
    # Toluene's methyl rotor is a hydrogen rotor, so its bond never becomes a central bond.
    mol = build_mol("Cc1ccccc1")
    torsions = torsion_constraints.find_heavy_atom_torsions(mol)
    assert all(mol.GetAtomWithIdx(idx).GetAtomicNum() > 1 for t in torsions for idx in t)
    methyl = next(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
    )
    assert methyl not in {atom for bond in central_bonds(torsions) for atom in bond}


def test_ring_and_biaryl_bonds_are_selected_like_xtb():
    # xtb keeps every bond whose central atoms are 2-coordinate, so ring, aromatic and
    # biaryl bonds are restrained too - unlike RDKit's strict "rotatable bond" definition.
    ring_torsions = torsion_constraints.find_heavy_atom_torsions(build_mol("c1ccccc1"))
    assert len(ring_torsions) == 6

    biaryl = build_mol("c1ccccc1-c1ccccc1")
    inter_ring = biaryl.GetSubstructMatch(Chem.MolFromSmarts("[cR1]-[cR1]"))
    assert frozenset(inter_ring) in central_bonds(
        torsion_constraints.find_heavy_atom_torsions(biaryl)
    )


def test_amide_torsion_is_kept():
    # The C-N bond is the only one with a heavy neighbour on both sides.
    mol = build_mol("CC(=O)NC")
    torsions = torsion_constraints.find_heavy_atom_torsions(mol)
    assert len(torsions) == 1
    (torsion,) = torsions
    carbon, _oxygen, nitrogen = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))
    assert set(torsion[1:3]) == {carbon, nitrogen}


def test_near_linear_torsions_are_skipped():
    # The C-C#N arrangement makes the torsion around the terminal C-C bond undefined.
    mol = build_mol("CCCC#N")
    torsions = torsion_constraints.find_heavy_atom_torsions(mol)
    nitrile_n = max(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    assert all(nitrile_n not in torsion for torsion in torsions)


def test_selection_is_independent_of_atom_order():
    """Unlike xtb's ``all torsions`` switch, the selection must not depend on ordering."""
    mol = build_mol("CC(=O)Nc1ccc(OCC(=O)N2CCOCC2)cc1")
    reversed_order = list(reversed(range(mol.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(mol, reversed_order)

    torsions = torsion_constraints.find_heavy_atom_torsions(mol)
    torsions_renumbered = torsion_constraints.find_heavy_atom_torsions(renumbered)

    back_map = {new: old for new, old in enumerate(reversed_order)}
    mapped = {frozenset(back_map[idx] for idx in torsion[1:3]) for torsion in torsions_renumbered}
    assert mapped == central_bonds(torsions)


def test_requires_a_conformation():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    with pytest.raises(ValueError, match="conformation"):
        torsion_constraints.find_heavy_atom_torsions(mol)
