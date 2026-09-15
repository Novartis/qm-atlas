from conftest import RESOURCES  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.file_interface import input_check

TEST_SDF = RESOURCES / "test_geometry.sdf"

CHEMBL508094 = "C[C@H](O)[C@H](C)[C@H](N)C(=O)O"
CHEMBL1741 = (
    "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)"
    "[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)"
    "[C@H](C)[C@@H](O)[C@]1(C)O"
)


def test_ignore_molecule():

    mol = Chem.MolFromSmiles(CHEMBL508094)
    assert input_check.ignore_molecule(mol) is None

    mol = Chem.MolFromSmiles(CHEMBL1741)
    assert input_check.ignore_molecule(mol, max_molecular_weight=700) is not None
    assert input_check.ignore_molecule(mol, max_molecular_weight=1000) is None
    assert input_check.ignore_molecule(mol, max_molecular_weight=700, override_checks=True) is None

    assert input_check.ignore_molecule(mol, max_heavy_atoms=50) is not None

    Chem.RemoveStereochemistry(mol)
    assert input_check.ignore_molecule(mol) is not None
