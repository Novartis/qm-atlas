import random

import numpy as np
from rdkit import Chem

from qm_atlas import utils

ERYTHROMYCIN_SMILES = (  # CHEMBL532
    "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)"
    "[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)"
    "[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"
)

IBUPROFEN_SMILES = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"  # CHEMBL521


def test_intervals_overlap():

    assert not utils.intervals_overlap((-np.inf, 1.5), (2, 3))
    assert not utils.intervals_overlap((1, 1.5), (2, 3))
    assert utils.intervals_overlap((-np.inf, 2), (2, 3))
    assert utils.intervals_overlap((1, 2), (2, 3))
    assert utils.intervals_overlap((-np.inf, 2.5), (2, 3))
    assert utils.intervals_overlap((1, 2.5), (2, 3))
    assert utils.intervals_overlap((1, 3), (2, 3))
    assert utils.intervals_overlap((1, 3.5), (2, 3))
    assert utils.intervals_overlap((2, 3), (2, 3))
    assert utils.intervals_overlap((2, 3.5), (2, 3))
    assert utils.intervals_overlap((3, 3.5), (2, 3))
    assert not utils.intervals_overlap((3.1, 3.5), (2, 3))


def test_are_equal():
    smiles_1 = "CCO"
    molobj_1 = Chem.MolFromSmiles(smiles_1)

    smiles_2 = "OCC"
    molobj_2 = Chem.MolFromSmiles(smiles_2)
    assert utils.are_equal(molobj_1, molobj_2)

    smiles_3 = "CCC"
    molobj_3 = Chem.MolFromSmiles(smiles_3)

    assert not utils.are_equal(molobj_1, molobj_3)


def test_has_macrocycle():
    benzene = Chem.MolFromSmiles("c1ccccc1")  # CHEMBL277500
    assert not utils.has_macrocycle(benzene)

    erythromycin = Chem.MolFromSmiles(ERYTHROMYCIN_SMILES)
    assert utils.has_macrocycle(erythromycin)


def test_as_float():
    assert utils.as_float("3.14") == 3.14
    assert utils.as_float("0") == 0.0
    assert utils.as_float("10") == 10.0
    assert utils.as_float("not a number") == "not a number"
    assert utils.as_float("") == ""
    assert utils.as_float(None) is None


def test_meta_func():
    # Define a simple function for testing purposes
    def add(x, y):
        return x + y

    # Test that meta_func correctly forwards its arguments to add
    assert utils.meta_func(add, (1, 2)) == 3


def test_restore_indices_large_mol():
    test_mol = Chem.AddHs(Chem.MolFromSmiles(ERYTHROMYCIN_SMILES))
    utils.store_atom_indices(test_mol)
    test_mol_edit = Chem.AddHs(Chem.RemoveHs(test_mol))
    utils.restore_atom_indices(test_mol_edit)
    assert utils.are_equal(test_mol, test_mol_edit)


def test_restore_indices_after_reordering():
    # first version. Use canonical rank (with hydrogens) for the reordering. That leads to
    # some hydrogen atoms being in between heavy atoms.
    ibuprofen = Chem.AddHs(Chem.MolFromSmiles(IBUPROFEN_SMILES))
    canon_ranks = Chem.CanonicalRankAtoms(ibuprofen)
    ibuprofen_ = Chem.RenumberAtoms(ibuprofen, canon_ranks)
    utils.store_atom_indices(ibuprofen_)

    ibuprofen_new = Chem.AddHs(Chem.RemoveHs(ibuprofen_))
    ibuprofen_new = utils.restore_atom_indices(ibuprofen_new)
    assert utils.are_equal(ibuprofen_, ibuprofen_new)

    matches = ibuprofen_.GetSubstructMatches(ibuprofen_new, uniquify=False, maxMatches=2**20)
    assert tuple(range(ibuprofen.GetNumAtoms())) in matches


def test_restore_indices_after_reordering_2():
    # second version. Put all hydrogens at the start.
    ibuprofen = Chem.AddHs(Chem.MolFromSmiles(IBUPROFEN_SMILES))
    new_order = list(range(ibuprofen.GetNumAtoms()))[::-1]
    ibuprofen_ = Chem.RenumberAtoms(ibuprofen, new_order)
    utils.store_atom_indices(ibuprofen_)

    ibuprofen_new = Chem.AddHs(Chem.RemoveHs(ibuprofen_))
    ibuprofen_new = utils.restore_atom_indices(ibuprofen_new)
    assert utils.are_equal(ibuprofen_, ibuprofen_new)

    matches = ibuprofen_.GetSubstructMatches(ibuprofen_new, uniquify=False, maxMatches=2**20)
    assert tuple(range(ibuprofen.GetNumAtoms())) in matches


def test_restore_indices_after_reordering_3():
    # third version. Use a random shuffle for the reordering. Use a fixed random seed for reproducibility.
    ibuprofen = Chem.AddHs(Chem.MolFromSmiles(IBUPROFEN_SMILES))
    random.seed(37)
    new_order = list(range(ibuprofen.GetNumAtoms()))
    random.shuffle(new_order)
    ibuprofen_ = Chem.RenumberAtoms(ibuprofen, new_order)
    utils.store_atom_indices(ibuprofen_)

    ibuprofen_new = Chem.AddHs(Chem.RemoveHs(ibuprofen_))
    ibuprofen_new = utils.restore_atom_indices(ibuprofen_new)
    assert utils.are_equal(ibuprofen_, ibuprofen_new)

    matches = ibuprofen_.GetSubstructMatches(ibuprofen_new, uniquify=False, maxMatches=2**20)
    assert tuple(range(ibuprofen.GetNumAtoms())) in matches
