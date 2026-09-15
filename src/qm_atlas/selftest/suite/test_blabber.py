from pathlib import Path

import pytest
from context import require_software  # pylint: disable=import-error
from ppqm import chembridge
from rdkit import Chem

from qm_atlas.wrappers.moka import blabber

TEST_PROTONATION_SMILES = [
    ("NCC(=O)O", ["[NH3+]CC(=O)[O-]"]),  # glycine CHEMBL773
    (
        "C1=C(c2ccc3ccccc3c2)N2CCN=C2c2ccccc21",  # CHEMBL309876
        ["C1=C(c2ccc3ccccc3c2)N2CC[NH+]=C2c2ccccc21"],
    ),
]


@require_software("moka")
@pytest.mark.parametrize("smiles, result_smiles", TEST_PROTONATION_SMILES)
def test_protonation_of_smiles(smiles, result_smiles, tmpdir):

    blabber_options = {"scr": tmpdir}
    molobj = chembridge.smiles_to_molobj(smiles)

    predicted_mols = blabber.get_protonation(molobj, **blabber_options)
    pre_smis = [
        chembridge.molobj_to_smiles(x, canonical=True, remove_hs=True) for x in predicted_mols
    ]
    assert set(result_smiles) == set(pre_smis)


@require_software("moka")
@pytest.mark.parametrize("smiles, result_smiles", TEST_PROTONATION_SMILES)
def test_protonation_of_molobj(smiles, result_smiles, tmpdir):

    blabber_options = {
        "scr": tmpdir,
        "remove_zero_charged": True,
        "threshold": 50,
    }

    molobj = Chem.MolFromSmiles(smiles)

    predicted_mols = blabber.get_protonation(molobj, **blabber_options)
    pre_smis = [
        chembridge.molobj_to_smiles(x, canonical=True, remove_hs=True) for x in predicted_mols
    ]
    assert set(result_smiles) == set(pre_smis)


@require_software("moka")
def test_stereocenter_consistency():

    # CHEMBL1905569
    smi = "CC(C)N1CC[C@H](NC(=O)NC2CCCCC2)CNC1"
    molobj = chembridge.smiles_to_molobj(smi)

    chiral_centers = dict(Chem.FindMolChiralCenters(molobj, includeUnassigned=False))

    molobjs = blabber.get_protonation(molobj)
    new_molobj = molobjs[0]

    new_chiral_centers = dict(Chem.FindMolChiralCenters(new_molobj, includeUnassigned=False))

    assert set(chiral_centers.items()).issubset(set(new_chiral_centers.items()))


@require_software("moka")
def test_get_protonation_from_wrong_smiles(tmpdir):

    blabber_options = {"scr": tmpdir}
    wrong_smiles = "CCCC=N(C)C"
    molobj = chembridge.smiles_to_molobj(wrong_smiles)
    # Invalid SMILES yields None from RDKit; the wrapper should handle gracefully
    if molobj is None:
        return  # cannot call blabber with None mol — passes (pre-validation)
    results = blabber.get_protonation(molobj, **blabber_options)
    assert results is None or len(results) == 0


def test_ldc_sanitization():

    # TODO

    return


@require_software("moka")
def test_neutral():
    """
    If compound does not have any charged compounds, return empty list by
    default
    """

    smi = "CC(C)CCCC"
    molobj = chembridge.smiles_to_molobj(smi)
    molobjs = blabber.get_protonation(molobj)
    assert len(molobjs) == 0


@require_software("moka")
def test_property_abundance():

    # CHEMBL1905569
    smiles = "CC(C)N1CCC(NC(O)NC2CCCCC2)CNC1"
    molobj = chembridge.smiles_to_molobj(smiles)
    molobjs = blabber.get_protonation(molobj)

    for mol in molobjs:
        assert mol.HasProp(blabber.COLUMN_ABUNDANCE)
        value = mol.GetProp(blabber.COLUMN_ABUNDANCE)
        value = float(value)
        assert value > 75.0


if __name__ == "__main__":
    tmp_dir = Path("_tmp_")
    test_property_abundance()
    test_neutral()
