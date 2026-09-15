import numpy as np
import pytest
from rdkit import Chem

from qm_atlas.wrappers.moka import process as pka

CHEMBL47248_SMILES = "OCCNCCNCCO"  # CHEMBL47248
NAME = "CHEMBL47248"
CENTER_INDICES = np.asarray([0, 9, 3, 6])
CENTER_TYPES = np.asarray(["a", "a", "b", "b"])

PKA_VALUES_LIST = [
    np.asarray([0.0, 2.0, 8.0, 10.0]),
    np.asarray([8.0, 10.0, 0.0, 2.0]),
    np.asarray([0.0, 0.0, 8.0, 10.0]),
    np.asarray([8.0, 8.0, 2.0, 2.0]),
]

EXPECTED_STATES_LIST = [
    {
        f"{NAME}_AA": "[O-]CCNCCNCC[O-]",
        f"{NAME}_A": "[O-]CCNCC[NH2+]CC[O-]",
        f"{NAME}_ZH": "[O-]CC[NH2+]CC[NH2+]CC[O-]",
        f"{NAME}_BH": "[O-]CC[NH2+]CC[NH2+]CCO",
        f"{NAME}_BHH": "OCC[NH2+]CC[NH2+]CCO",
    },
    {
        f"{NAME}_AA": "[O-]CCNCCNCC[O-]",
        f"{NAME}_A": "[O-]CCNCCNCCO",
        NAME: "OCCNCCNCCO",
        f"{NAME}_BH": "OCCNCC[NH2+]CCO",
        f"{NAME}_BHH": "OCC[NH2+]CC[NH2+]CCO",
    },
]

EXPECTED_SINGLE_CHARGE_STATES = {
    f"{NAME}_A": ["OCCNCCNCC[O-]"],
    NAME: ["OCCNCCNCCO"],
    f"{NAME}_BH": ["OCCNCC[NH2+]CCO"],
}


def canonicalize_smiles(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


@pytest.mark.parametrize("pka_values", PKA_VALUES_LIST[:3])
def test_get_pka_centers(pka_values: list[float]):

    idcs = np.argsort(pka_values)
    center_types_sorted = CENTER_TYPES[idcs]
    center_indices_sorted = CENTER_INDICES[idcs]
    pka_values_sorted = pka_values[idcs]

    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    assert len(pka_centers) == len(CENTER_TYPES)

    pka_centers_sorted = sorted(pka_centers)
    pka_values_from_centers = np.asarray(
        [pka_center.pka_value for pka_center in pka_centers_sorted]
    )
    assert np.allclose(pka_values_from_centers, pka_values_sorted)

    center_indices_ = np.asarray([pka_center.index for pka_center in pka_centers_sorted])
    assert np.allclose(center_indices_, center_indices_sorted)

    center_types_ = np.asarray([pka_center.center_type for pka_center in pka_centers_sorted])
    assert (center_types_ == center_types_sorted).all()


@pytest.mark.parametrize(
    "pka_values, expected_states",
    [
        [PKA_VALUES_LIST[0], EXPECTED_STATES_LIST[0]],
        [PKA_VALUES_LIST[1], EXPECTED_STATES_LIST[1]],
    ],
)
def test_generate_ionic_species(pka_values, expected_states):

    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)
    pka_calculation_info = pka.generate_ionic_species(mol, pka_centers)

    assert len(pka_calculation_info.name_to_mol) == len(expected_states)
    assert len(pka_calculation_info.pka_centers) == len(pka_values)

    for name, (mol, _) in pka_calculation_info.name_to_mol.items():
        assert name in expected_states
        expected_smiles = expected_states[name]
        assert canonicalize_smiles(Chem.MolToSmiles(Chem.RemoveHs(mol))) == canonicalize_smiles(
            expected_smiles
        )
        expected_mol = Chem.MolFromSmiles(expected_smiles)
        if not tuple(range(expected_mol.GetNumAtoms())) in mol.GetSubstructMatches(expected_mol):
            print(name)
        assert tuple(range(expected_mol.GetNumAtoms())) in mol.GetSubstructMatches(expected_mol)


def test_generate_ionic_species_thresholds():

    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)

    # weak acidic and basic centers
    pka_values = np.asarray([7.8, 11.2, 4.2, 6.5])
    lower_threshold = 6.4
    upper_threshold = 8.4
    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)

    pka_calculation_info = pka.generate_ionic_species(
        mol,
        pka_centers,
        lower_ph_threshold=lower_threshold,
        upper_ph_threshold=upper_threshold,
    )

    assert len(pka_calculation_info.name_to_mol) == 3
    assert len(pka_calculation_info.pka_centers) == 2


def test_generate_ionic_species_thresholds_2():

    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)

    # strong acidic and basic centers, neutral state will be double zwitterion
    pka_values = np.asarray([4.2, 6.5, 8.2, 11.2])
    lower_threshold = 6.4
    upper_threshold = 8.4
    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    pka_calculation_info = pka.generate_ionic_species(
        mol,
        pka_centers,
        lower_ph_threshold=lower_threshold,
        upper_ph_threshold=upper_threshold,
    )
    print(pka_calculation_info.name_to_mol.keys())
    assert len(pka_calculation_info.name_to_mol) == 5
    num_mols_to_run = [mol for mol, expand in pka_calculation_info.name_to_mol.values() if expand]
    assert len(num_mols_to_run) == 3

    assert f"{NAME}_ZH" in pka_calculation_info.name_to_mol

    neutral_state, _ = pka_calculation_info.name_to_mol[f"{NAME}_ZH"]
    assert sum([abs(atom.GetFormalCharge()) for atom in neutral_state.GetAtoms()]) == 4


def test_generate_ionic_species_thresholds_3():

    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)

    # only one relevant acidic center, outside the range
    pka_values = np.asarray([4.2, 10.5, 1.1, 2.2])
    lower_threshold = 6.4
    upper_threshold = 8.4
    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    pka_calculation_info = pka.generate_ionic_species(
        mol,
        pka_centers,
        lower_ph_threshold=lower_threshold,
        upper_ph_threshold=upper_threshold,
    )

    assert len(pka_calculation_info.name_to_mol) == 2
    num_mols_to_run = [mol for mol, expand in pka_calculation_info.name_to_mol.values() if expand]
    assert len(num_mols_to_run) == 1

    assert f"{NAME}_A" in pka_calculation_info.name_to_mol

    acid_state, _ = pka_calculation_info.name_to_mol[f"{NAME}_A"]
    assert Chem.GetFormalCharge(acid_state) == -1
    assert sum([abs(atom.GetFormalCharge()) for atom in acid_state.GetAtoms()]) == 1


def test_generate_ionic_species_thresholds_4():

    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)

    # only one relevant basic center, outside the range
    pka_values = np.asarray([10.2, 10.5, 1.1, 8.5])
    lower_threshold = 6.4
    upper_threshold = 8.4
    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    pka_calculation_info = pka.generate_ionic_species(
        mol,
        pka_centers,
        lower_ph_threshold=lower_threshold,
        upper_ph_threshold=upper_threshold,
    )

    assert len(pka_calculation_info.name_to_mol) == 2
    num_mols_to_run = [mol for mol, expand in pka_calculation_info.name_to_mol.values() if expand]
    assert len(num_mols_to_run) == 1

    assert f"{NAME}_BH" in pka_calculation_info.name_to_mol

    base_state, _ = pka_calculation_info.name_to_mol[f"{NAME}_BH"]
    assert Chem.GetFormalCharge(base_state) == 1
    assert sum([abs(atom.GetFormalCharge()) for atom in base_state.GetAtoms()]) == 1


@pytest.mark.parametrize(
    "pka_values, expected_states",
    [
        [PKA_VALUES_LIST[0], EXPECTED_SINGLE_CHARGE_STATES],
        [PKA_VALUES_LIST[1], EXPECTED_SINGLE_CHARGE_STATES],
    ],
)
def test_generate_single_charge_ions(pka_values, expected_states):

    pka_centers = pka.get_pka_centers_from_arrays(CENTER_TYPES, CENTER_INDICES, pka_values)
    mol = Chem.MolFromSmiles(CHEMBL47248_SMILES)
    mol.SetProp("_Name", NAME)
    pka_calculation_info = pka.generate_single_charge_ions(mol, pka_centers)

    assert len(pka_calculation_info.name_to_mol) == len(expected_states)
    assert len(pka_calculation_info.pka_centers) == len(pka_values)

    for name, states_list in expected_states.items():
        if "_A" in name:
            charge = -1
        elif "_BH" in name:
            charge = 1
        else:
            charge = 0
        compare_mols = [
            mol
            for mol, _ in pka_calculation_info.name_to_mol.values()
            if Chem.GetFormalCharge(mol) == charge
        ]
        compare_smis = [
            canonicalize_smiles(Chem.MolToSmiles(Chem.RemoveHs(mol))) for mol in compare_mols
        ]
        # note that due to the symmetry of the test molecule, the single charge ions generated for each of the two
        # acidic or basic centers are the same; and we only want to do the cosmo calculations for one of them.
        assert len(compare_smis) == len(states_list)

        for smi in states_list:
            assert canonicalize_smiles(smi) in compare_smis
