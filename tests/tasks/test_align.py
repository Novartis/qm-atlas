"""Tests for the align task module."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.tasks import align


def _make_two_conformer_mol():
    """Return a molecule with two distinct embedded conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=0xF00D)
    return mol


def test_registries_are_consistent():
    assert set(align.OPTIONS_CLASS_REGISTRY) == set(align.ALIGNMENT_FUNCTIONS)
    assert align.RDKIT_NAME in align.OPTIONS_CLASS_REGISTRY


def test_deserialize_alignment_config_via_dict():
    """A YAML dict maps to the right backend via the registry."""
    cfg = align.deserialize_alignment_config({"backend": "rdkit"})
    assert isinstance(cfg, align.RdkitAlignmentOptions)
    # An already-instantiated config passes through unchanged.
    assert align.deserialize_alignment_config(cfg) is cfg


def test_deserialize_alignment_config_defaults_to_rdkit():
    cfg = align.deserialize_alignment_config({})
    assert isinstance(cfg, align.RdkitAlignmentOptions)


def test_deserialize_alignment_config_unknown_backend_raises():
    with pytest.raises(ValueError):
        align.deserialize_alignment_config({"backend": "does_not_exist"})


def test_align_to_reference_returns_rmsd_per_conformer():
    mol = _make_two_conformer_mol()
    reference = Chem.Mol(mol)  # same graph, use conformer 0 as the reference

    aligned_mol, rmsd_dict = align.align_to_reference(
        mol, reference, align.RdkitAlignmentOptions(), ref_cid=0
    )

    conf_ids = [c.GetId() for c in aligned_mol.GetConformers()]
    assert set(rmsd_dict) == set(conf_ids)
    assert all(rmsd >= 0 for rmsd in rmsd_dict.values())


def test_align_to_reference_preserves_properties():
    mol = _make_two_conformer_mol()
    mol.SetProp("origin", "unit-test")
    reference = Chem.Mol(mol)

    aligned_mol, _ = align.align_to_reference(mol, reference, align.RdkitAlignmentOptions())

    assert aligned_mol.HasProp("origin")
    assert aligned_mol.GetProp("origin") == "unit-test"
    assert aligned_mol.GetNumAtoms() == mol.GetNumAtoms()


def test_dispatch_selects_backend_by_name():
    mol = _make_two_conformer_mol()
    reference = Chem.Mol(mol)

    # A single-conformer probe aligned to itself must have ~zero RMSD.
    single = Chem.Mol(mol)
    single.RemoveAllConformers()
    single.AddConformer(mol.GetConformer(0), assignId=True)

    _, rmsd_dict = align.align_to_reference(single, reference, align.RdkitAlignmentOptions())
    assert rmsd_dict[0] == pytest.approx(0.0, abs=1e-4)


def test_rdkit_options_reject_unknown_fields():
    with pytest.raises(ValueError):
        align.RdkitAlignmentOptions(nonexistent_option=True)  # type: ignore
