from pathlib import Path

import numpy as np
import pytest
from context import TEST_CONF_GEN_OPTIONS, require_software  # pylint: disable=import-error
from ppqm import chembridge
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.workflows import fast_conformers

CHEMBL508094 = "C[C@H](O)[C@H](C)[C@H](N)C(=O)O"
CHEMBL362153 = "C/C=C1/C[C@@H](C)[C@@](C)(O)C(=O)OCC2=CCN3CC[C@@H](OC1=O)[C@@H]23"

TEST_SMILES_CONF_GEN = [CHEMBL508094, CHEMBL362153]


@pytest.mark.parametrize("smiles", TEST_SMILES_CONF_GEN)
@require_software("xtb")
def test_generate_conformers(smiles, tmpdir):
    """Test the main fast_conformers.generate_fast_conformers() method"""
    molobj = Chem.MolFromSmiles(smiles)
    molobj.SetProp("_Name", "test_molecule")
    molobj.SetProp("Test", "test_value")
    molobj = fast_conformers.generate_fast_conformers(
        molobj,
        scr=Path(str(tmpdir)),
        show_progress=False,
        conformer_generation_options=TEST_CONF_GEN_OPTIONS,
    )

    assert molobj is not None
    assert molobj.GetNumConformers() > 0
    assert molobj.GetProp("_Name") == "test_molecule"
    assert molobj.GetProp("Test") == "test_value"


def _mol_with_conformers(n_confs: int) -> Chem.Mol:
    """A molecule carrying `n_confs` embedded conformers with ids 0..n_confs-1."""
    molobj = Chem.AddHs(Chem.MolFromSmiles(CHEMBL508094))
    AllChem.EmbedMultipleConfs(molobj, numConfs=n_confs, randomSeed=1)
    assert molobj.GetNumConformers() == n_confs
    return molobj


def test_reindex_conformers_is_a_no_op_when_ids_are_contiguous():
    """An untouched ensemble is returned unchanged, without copying."""
    molobj = _mol_with_conformers(3)
    assert fast_conformers.reindex_conformers(molobj) is molobj


def test_reindex_conformers_closes_gaps_left_by_failed_optimizations():
    """A gap in the conformer ids is closed, preserving order and coordinates."""
    molobj = _mol_with_conformers(3)
    kept = [molobj.GetConformer(i).GetPositions() for i in (0, 2)]
    molobj.RemoveConformer(1)  # what optimize_mol leaves behind for a failed conformer
    assert [conf.GetId() for conf in molobj.GetConformers()] == [0, 2]

    reindexed = fast_conformers.reindex_conformers(molobj)

    assert [conf.GetId() for conf in reindexed.GetConformers()] == [0, 1]
    for position, expected in enumerate(kept):
        assert np.allclose(reindexed.GetConformer(position).GetPositions(), expected)
    # the input is left alone
    assert [conf.GetId() for conf in molobj.GetConformers()] == [0, 2]


def test_reindexed_conformers_are_addressable_by_ppqm():
    """The gap makes chembridge.get_sasa raise; reindexing is what fixes it."""
    molobj = _mol_with_conformers(3)
    molobj.RemoveConformer(1)

    with pytest.raises(ValueError):
        chembridge.get_sasa(molobj, extra_radius=1.0)

    sasas = chembridge.get_sasa(fast_conformers.reindex_conformers(molobj), extra_radius=1.0)
    assert len(sasas) == 2
    assert all(sasa > 0.0 for sasa in sasas)
