import numpy as np
from context import require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.wrappers.jaguar import bond_dissociation, qm_descriptors

TEST_MOLBLOCK = """test molecule
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.1279   -0.0772   -0.0040 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0216   -1.1047    0.8417 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.3965    1.4141    0.9229 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -1.4841    0.0407   -1.0993 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9814   -0.2728   -0.6612 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
"""

EXP_BDE = {
    0: 102.76,
}

EXP_PROPERTIES = {
    "s_j_Atom_Fukui_Index_f_NN_HOMO": np.array([0.0069, 0.0081, 0.175, 0.8084, 0.0017]),
    "s_j_Lowdin_Atom_Charge": np.array([-0.20808, -0.13313, 0.00968, 0.1492, 0.18233]),
    "s_j_Atom_NMR_Isotropic_Shielding": np.array([69.6832, 264.2899, 563.9533, 46.6375, 24.7801]),
}


@require_software("jaguar")
def test_hydrogen_abstraction():
    test_mol = Chem.rdmolfiles.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    abst_energies_res = bond_dissociation.get_hydrogen_abstraction_energies(
        test_mol, 0, "C", n_cores=2
    )

    abst_energies = abst_energies_res[bond_dissociation.PROPERTY_NAME].get_property_value()
    assert abst_energies.keys() == EXP_BDE.keys()

    for key, abst_eng_exp in EXP_BDE.items():

        assert key in abst_energies
        abst_eng = abst_energies[key]
        assert np.abs(abst_eng - abst_eng_exp) <= 0.01


@require_software("jaguar")
def test_descriptors():
    test_mol = Chem.rdmolfiles.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    properties_dict = qm_descriptors.calculate_descriptors(
        test_mol, 0, ["fukui", "lowdin", "nmr"], n_cores=2
    )

    for key, prop_exp in EXP_PROPERTIES.items():
        assert key in properties_dict
        prop_calc = properties_dict[key].get_property_value()
        assert np.allclose(prop_calc, prop_exp, atol=1.0e-3)
