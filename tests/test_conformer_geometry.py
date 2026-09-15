from itertools import combinations

import numpy as np
from conftest import RESOURCES
from rdkit import Chem, Geometry
from rdkit.Chem import rdMolAlign
from rdkit.Numerics import rdAlignment

from qm_atlas.tasks import conformer_generation
from qm_atlas.tasks.utils import conformer_geometry

TEST_SDF = RESOURCES / "test_geometry.sdf"


def get_molecules() -> list[Chem.Mol]:
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        mols = [mol for mol in supplier]
    return mols


def get_molecule() -> Chem.Mol:
    mol = None
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        for add_mol in supplier:
            mol = conformer_generation.join_conformers(mol, add_mol)
    return mol


def test_join_molecules():
    mols = get_molecules()
    mol, _ = conformer_geometry.join_molecules(mols[0], mols[1])
    assert mol.GetNumConformers() == 2

    mol_1 = get_molecule()
    assert mol_1.GetNumConformers() == 3
    mol, _ = conformer_geometry.join_molecules(mols[0], mol_1)
    assert mol.GetNumConformers() == 4

    mol, _ = conformer_geometry.join_molecules(mols[0], mol_1, ref_cids=[2])
    assert mol.GetNumConformers() == 2


def test_align_conformers():
    mol = get_molecule()

    rmsd_dict = conformer_geometry.align_conformers(mol)
    rmsd_01 = rdMolAlign.CalcRMS(mol, mol, prbId=0, refId=1)
    np.testing.assert_almost_equal(rmsd_dict[1], rmsd_01, decimal=6)

    conformer_0 = mol.GetConformer(0)
    conformer_1 = mol.GetConformer(1)
    num_atoms = mol.GetNumAtoms()
    coords_0 = [conformer_0.GetAtomPosition(idx) for idx in range(num_atoms)]
    coords_1 = [conformer_1.GetAtomPosition(idx) for idx in range(num_atoms)]
    ssr, transf = rdAlignment.GetAlignmentTransform(coords_0, coords_1)
    rmsd_ = np.sqrt(ssr / num_atoms)

    np.testing.assert_almost_equal(rmsd_01, rmsd_, decimal=6)

    assert np.allclose(transf, np.diag((1, 1, 1, 1)), atol=1.0e-4)


def test_get_duplicate():
    mol = get_molecule()
    duplicate_idcs = conformer_geometry.get_duplicate_idcs(mol, 0.5)
    assert duplicate_idcs == set([2])
    duplicate_idcs = conformer_geometry.get_duplicate_idcs(mol, 0.5, method="exact")
    assert duplicate_idcs == set([2])

    duplicate_idcs = conformer_geometry.get_duplicate_idcs(mol, 0.2)
    assert len(duplicate_idcs) == 0


def test_rmsd_calculation():
    test_mol = get_molecule()

    h_rmsd = conformer_geometry.HydrogenConformerRMSD(test_mol)
    rmsd_mat_list = h_rmsd.get_rmsd_matrix()
    conf_ids = [conf.GetId() for conf in test_mol.GetConformers()]
    for id_1, id_2 in combinations(conf_ids, 2):
        rmsd = rdMolAlign.GetBestRMS(test_mol, test_mol, prbId=id_1, refId=id_2)
        list_idx = h_rmsd.get_list_idx((id_1, id_2))
        assert (id_1, id_2) == h_rmsd.get_conf_idcs(list_idx)
        assert list_idx == conformer_geometry.mat_to_list_idx((id_1, id_2))
        assert (id_1, id_2) == conformer_geometry.list_to_mat_idx(list_idx)

        assert np.isclose(rmsd, rmsd_mat_list[list_idx], atol=1.0e-4)

        rmsd_again, *_ = h_rmsd.get_conformer_rmsd((id_1, id_2))

        assert np.isclose(rmsd, rmsd_again, atol=1.0e-7)


def get_terminal_heavy_atom(mol: Chem.Mol) -> int:
    """index of a heavy atom with a single heavy neighbor and no hydrogens attached"""
    for atom in mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if atom.GetSymbol() != "H" and len(neighbors) == 1 and neighbors[0].GetSymbol() != "H":
            return atom.GetIdx()
    raise AssertionError("test molecule has no terminal heavy atom")


def get_remote_heavy_atom(mol: Chem.Mol, atom_idx: int) -> int:
    """index of the heavy atom which is furthest from atom_idx in the bond table"""
    topological_distances = Chem.GetDistanceMatrix(mol)[atom_idx]
    heavy_idcs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
    return max(heavy_idcs, key=lambda idx: topological_distances[idx])


def test_geometry_consistency_intact():
    for mol in get_molecules():
        assert conformer_geometry.check_geometry_consistency(mol, 0) == []
        assert conformer_geometry.sanity_check(mol, 0)


def test_geometry_consistency_broken_bond():
    test_mol = get_molecules()[0]
    conformer = test_mol.GetConformer(0)
    positions = conformer.GetPositions()

    # pull a terminal atom away from its parent along the bond axis
    atom_idx = get_terminal_heavy_atom(test_mol)
    parent_idx = test_mol.GetAtomWithIdx(atom_idx).GetNeighbors()[0].GetIdx()
    new_position = positions[parent_idx] + 2.5 * (positions[atom_idx] - positions[parent_idx])
    conformer.SetAtomPosition(atom_idx, Geometry.Point3D(*new_position))

    violations = conformer_geometry.check_geometry_consistency(test_mol, 0)
    assert not conformer_geometry.sanity_check(test_mol, 0)
    assert violations == ["stretched_bond"]


def test_geometry_consistency_new_bond():
    test_mol = get_molecules()[0]
    conformer = test_mol.GetConformer(0)
    positions = conformer.GetPositions()

    # move a terminal atom next to a heavy atom it is not bonded to
    atom_idx = get_terminal_heavy_atom(test_mol)
    partner_idx = get_remote_heavy_atom(test_mol, atom_idx)
    direction = positions[atom_idx] - positions[partner_idx]
    direction /= np.linalg.norm(direction)
    conformer.SetAtomPosition(
        atom_idx, Geometry.Point3D(*(positions[partner_idx] + 1.4 * direction))
    )

    violations = conformer_geometry.check_geometry_consistency(test_mol, 0)
    assert not conformer_geometry.sanity_check(test_mol, 0)
    assert "unexpected_contact" in violations

    # the contact test can be switched off, the broken bond is still detected
    violations = conformer_geometry.check_geometry_consistency(test_mol, 0, contact_scale=None)
    assert "stretched_bond" in violations
    assert "unexpected_contact" not in violations
