from context import TEST_CONF_GEN_OPTIONS, require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.workflows import score_tautomers

TEST_SMILES = "Nc1nc2[nH]cnc2c(=O)[nH]1"  # CHEMBL219568


@require_software("unicon", "xtb", "turbomole", "cosmotherm")
def test_tautomer_generation():
    test_mol = Chem.MolFromSmiles(TEST_SMILES)
    tautomers = score_tautomers.generate_tautomers(
        test_mol, conformer_generation_options=TEST_CONF_GEN_OPTIONS
    )

    assert len(tautomers) == 4
