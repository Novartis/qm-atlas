import pytest
from context import require_software  # pylint: disable=import-error
from ppqm import chembridge
from rdkit import Chem

from qm_atlas.wrappers import unicon

TEST_SMILES = [
    (
        "Nc1nc2[nH]cnc2c(=O)[nH]1",  # CHEMBL219568
        [
            "Nc1nc(=O)c2[nH]cnc2[nH]1",
            "Nc1nc2nc[nH]c2c(=O)[nH]1",
            "Nc1nc(=O)c2nc[nH]c2[nH]1",
            "Nc1nc2[nH]cnc2c(=O)[nH]1",
        ],
    ),
]


@require_software("unicon")
@pytest.mark.parametrize("smiles, result_smiles", TEST_SMILES)
def test_tautomer_smiles(smiles, result_smiles, tmpdir):

    scr = str(tmpdir)
    molobj = Chem.MolFromSmiles(smiles)

    tau_mols = unicon.get_tautomers(molobj, scr=scr)
    tau_smiles = [chembridge.molobj_to_smiles(m, canonical=True) for m in tau_mols]
    assert set(result_smiles) == set(tau_smiles)


@require_software("unicon")
def test_keep_stereocenters(tmpdir):

    scr = tmpdir

    # estradiol, CHEMBL135
    smi = "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O"
    molobj = Chem.MolFromSmiles(smi)

    tau_mols = unicon.get_tautomers(molobj, scr=scr)
    tau_smiles = [chembridge.molobj_to_smiles(m, canonical=True) for m in tau_mols]

    assert smi == tau_smiles[0]


@require_software("unicon")
@pytest.mark.parametrize("smiles, result_smiles", TEST_SMILES)
def test_tautomer_mol(smiles, result_smiles, tmpdir):

    unicon_options = {"scr": tmpdir}

    molobj = Chem.MolFromSmiles(smiles)

    # TODO: return signature changes depending on input type. Not a fan...
    tautomers = unicon.get_tautomers(molobj, **unicon_options)
    assert isinstance(tautomers, list)
    assert isinstance(tautomers[0], Chem.Mol)

    tautomers = unicon.get_tautomers(molobj, **unicon_options)
    assert isinstance(tautomers, list)
    assert isinstance(tautomers[0], Chem.Mol)


@require_software("unicon")
def test_get_tautomers_error(tmpdir):

    unicon_options = {"scr": tmpdir}

    wrong_type = {"hello": "wrong"}

    # get type error
    with pytest.raises(TypeError):
        unicon.get_tautomers(wrong_type, **unicon_options)
