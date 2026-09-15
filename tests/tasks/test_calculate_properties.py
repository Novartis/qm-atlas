"""Tests for calculate_properties module"""
from functools import partial
from unittest import mock

import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.constants import COSMO_OUTPUT_KEY
from qm_atlas.tasks import calculate_properties
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.tasks import conformer_generation
from qm_atlas.tasks.calculate_properties import PropertyCalculatorOptions

TEST_SDF = RESOURCES / "test_geometry.sdf"
MOCK_COSMO_OUTPUT = """
Not a real cosmo file but has some text you expect there:
$cosmo_energy
$segment_information
"""


class MockConfig(PropertyCalculatorOptions):
    """Mock config class for testing calculators."""

    backend: str

    def get_property_prefix(self) -> str:
        return "mock_"

    def get_property_names(self) -> dict[str, str]:
        return {}

    def get_property_types(self) -> dict[str, type]:
        return {}


def create_mock_config(calculator_name: str) -> MockConfig:
    """Create a mock config object with the specified calculator name."""
    return MockConfig(backend=calculator_name)


def get_molecules() -> list[Chem.Mol]:
    """Load test molecules from SDF file"""
    mols = []
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        mols = [mol for mol in supplier]
    return mols


def get_molecule() -> Chem.Mol:
    """Load first test molecule"""
    return get_molecules()[0]


def get_molecule_with_conformers() -> Chem.Mol:
    """Load molecule with multiple conformers"""
    mol = None
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        for add_mol in supplier:
            mol = conformer_generation.join_conformers(mol, add_mol)
    return mol


def create_mock_properties(
    energy: float = -130.2,
    include_cosmo: bool = False,
) -> dict[str, calculated_properties.Property]:
    """Create mock calculated properties"""
    properties: dict[str, calculated_properties.Property] = {
        "energy": calculated_properties.ScalarProperty(energy),
        "weight": calculated_properties.ScalarProperty(0.341),
    }
    if include_cosmo:
        properties[COSMO_OUTPUT_KEY] = calculated_properties.StringProperty(MOCK_COSMO_OUTPUT)
    return properties


def mock_calculator_function(
    mol: Chem.Mol,
    conf_id: int,
    ids_to_fail: set[int] | None = None,
    energy: float = -130.2,
    include_cosmo: bool = False,
    **kwargs,
) -> dict[str, calculated_properties.Property]:
    """Mock calculator function for property calculations."""
    del mol  # unused but API-required

    if ids_to_fail is None:
        ids_to_fail = set()

    del kwargs  # unused but API-required

    if conf_id in ids_to_fail:
        raise RuntimeError(f"Calculation failed for conformer ID {conf_id}")

    return create_mock_properties(energy=energy, include_cosmo=include_cosmo)


# Tests for calculate_property function
def test_calculate_property_success():
    """Test successful property calculation with default calculator"""
    mol = get_molecule()
    conf_id = 0

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator_function,
        },
    ):
        result_props = calculate_properties.calculate_property(
            mol,
            conf_id,
            config=create_mock_config("test_calculator"),
        )

        assert result_props is not None
        assert "energy" in result_props
        assert "weight" in result_props
        assert result_props["energy"].get_property_value() == -130.2


def test_calculate_property_with_kwargs():
    """Test that kwargs are passed to the calculator function"""
    mol = get_molecule()
    conf_id = 0
    custom_energy = -150.5

    mock_config = create_mock_config("test_calculator")
    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock.Mock(
                return_value=create_mock_properties(energy=custom_energy)
            ),
        },
    ):
        result_props = calculate_properties.calculate_property(
            mol,
            conf_id,
            config=mock_config,
        )

        assert result_props is not None
        assert result_props["energy"].get_property_value() == custom_energy


def test_calculate_property_failure_with_suppress_errors():
    """Test that None is returned on failure when suppress_errors=True"""
    mol = get_molecule()
    conf_id = 0

    mock_calculator = partial(mock_calculator_function, ids_to_fail={conf_id})

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_props = calculate_properties.calculate_property(
            mol,
            conf_id,
            config=create_mock_config("test_calculator"),
            suppress_errors=True,
        )

        assert result_props is None


def test_calculate_property_failure_without_suppress_errors():
    """Test that RuntimeError is raised on failure when suppress_errors=False"""
    mol = get_molecule()
    conf_id = 0

    mock_calculator = partial(mock_calculator_function, ids_to_fail={conf_id}, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        with pytest.raises(RuntimeError):
            calculate_properties.calculate_property(
                mol,
                conf_id,
                config=create_mock_config("test_calculator"),
                suppress_errors=False,
            )


def test_calculate_property_unknown_calculator():
    """Test ValueError for unknown calculator name"""
    mol = get_molecule()
    conf_id = 0

    with pytest.raises(ValueError, match="Unknown property calculator"):
        calculate_properties.calculate_property(
            mol,
            conf_id,
            config=create_mock_config("unknown_calculator"),
        )


def test_calculate_property_with_cosmo_output():
    """Test property calculation that includes COSMO output"""
    mol = get_molecule()
    conf_id = 0

    mock_calculator = partial(mock_calculator_function, identifier="test", include_cosmo=True)

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_props = calculate_properties.calculate_property(
            mol,
            conf_id,
            config=create_mock_config("test_calculator"),
        )

        assert result_props is not None
        assert COSMO_OUTPUT_KEY in result_props


# Tests for calculate_property_conformer function
def test_calculate_property_conformer_success():
    """Test successful conformer property calculation"""
    mol = get_molecule_with_conformers()
    conformer = mol.GetConformer(1)

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_props = calculate_properties.calculate_property_conformer(
            conformer,
            config=create_mock_config("test_calculator"),
        )

        assert result_props is not None
        assert "energy" in result_props


def test_calculate_property_conformer_failure_suppress():
    """Test conformer calculation failure with suppress_errors=True"""
    mol = get_molecule_with_conformers()
    conformer = mol.GetConformer(0)

    mock_calculator = partial(mock_calculator_function, ids_to_fail={0}, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_props = calculate_properties.calculate_property_conformer(
            conformer,
            config=create_mock_config("test_calculator"),
            suppress_errors=True,
        )

        assert result_props is None


def test_calculate_property_conformer_failure_no_suppress():
    """Test conformer calculation failure with suppress_errors=False"""
    mol = get_molecule_with_conformers()
    conformer = mol.GetConformer(0)

    mock_calculator = partial(mock_calculator_function, ids_to_fail={0}, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        with pytest.raises(RuntimeError):
            calculate_properties.calculate_property_conformer(
                conformer,
                config=create_mock_config("test_calculator"),
                suppress_errors=False,
            )


# Tests for calculate_property_mol function
def test_calculate_property_mol_single_conformer():
    """Test property calculation for molecule with single conformer"""
    mol = get_molecule()

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mol(
            mol,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == 1
        assert result_list[0] is not None
        assert "energy" in result_list[0]


def test_calculate_property_mol_multiple_conformers():
    """Test property calculation for molecule with multiple conformers"""
    mol = get_molecule_with_conformers()
    num_confs = mol.GetNumConformers()

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mol(
            mol,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == num_confs
        for i, props in enumerate(result_list):
            assert props is not None, f"Expected properties for conformer {i}"
            assert "energy" in props


def test_calculate_property_mol_with_n_cores():
    """Test property calculation with multiple CPU cores"""
    mol = get_molecule_with_conformers()
    num_confs = mol.GetNumConformers()

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mol(
            mol,
            config=create_mock_config("test_calculator"),
            n_cores=4,
        )

        assert result_list is not None
        assert len(result_list) == num_confs


def test_calculate_property_mol_partial_failure():
    """Test handling of partial conformer calculation failure"""
    mol = get_molecule_with_conformers()
    num_confs = mol.GetNumConformers()
    assert num_confs > 1, "Test requires molecule with multiple conformers"

    # Mock calculator that fails for conformer ID 1
    mock_calculator = partial(mock_calculator_function, ids_to_fail={1})

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mol(
            mol,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == num_confs
        # Check that conformer 1 failed (returns None)
        assert result_list[1] is None
        # Check that other conformers succeeded
        for i in range(num_confs):
            if i != 1:
                assert result_list[i] is not None


# Tests for calculate_property_mols function
def test_calculate_property_mols_single_molecule():
    """Test property calculation for single molecule with multiple conformers"""
    mol = get_molecule_with_conformers()
    num_confs = mol.GetNumConformers()

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mols(
            [mol],
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == 1
        assert len(result_list[0]) == num_confs
        for props in result_list[0]:
            assert props is not None
            assert "energy" in props


def test_calculate_property_mols_multiple_molecules():
    """Test property calculation for multiple molecules"""
    mols = get_molecules()[:2]  # Get two molecules
    num_confs_per_mol = [mol.GetNumConformers() for mol in mols]

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mols(
            mols,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == len(mols)
        for mol_idx, (num_confs, props_list) in enumerate(zip(num_confs_per_mol, result_list)):
            assert len(props_list) == num_confs, (
                f"Molecule {mol_idx}: expected {num_confs} results, " f"got {len(props_list)}"
            )


def test_calculate_property_mols_with_n_cores():
    """Test property calculation with parallel processing"""
    mols = get_molecules()[:2]

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mols(
            mols,
            config=create_mock_config("test_calculator"),
            n_cores=4,
        )

        assert result_list is not None
        assert len(result_list) == len(mols)


def test_calculate_property_mols_with_progress():
    """Test property calculation with progress bar"""
    mols = get_molecules()[:2]

    mock_calculator = partial(mock_calculator_function, identifier="test")

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mols(
            mols,
            config=create_mock_config("test_calculator"),
            show_progress=True,
        )

        assert result_list is not None


def test_calculate_property_mols_partial_failure():
    """Test handling of partial conformer calculation failure across molecules"""
    mol_joined = get_molecule_with_conformers()
    # Split into separate molecules to test the behavior with multiple molecules
    mols = []
    for conf in mol_joined.GetConformers():
        new_mol = Chem.Mol(mol_joined)
        # Remove all conformers except the current one
        conf_ids_to_remove = [
            c.GetId() for c in mol_joined.GetConformers() if c.GetId() != conf.GetId()
        ]
        for cid in sorted(conf_ids_to_remove, reverse=True):
            new_mol.RemoveConformer(cid)
        mols.append(new_mol)

    # Use conformer ID 0 for the first molecule (which will fail)
    mock_calculator = partial(mock_calculator_function, ids_to_fail={0})

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mols(
            mols,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        assert len(result_list) == len(mols)
        # First molecule should have a failed conformer
        assert result_list[0][0] is None
        # All other molecules should succeed
        for mol_idx in range(1, len(result_list)):
            for props in result_list[mol_idx]:
                assert props is not None


def test_calculate_property_mols_unknown_calculator():
    """Test ValueError for unknown calculator in mols calculation"""
    mol = get_molecule()

    with pytest.raises(ValueError, match="Unknown property calculator"):
        calculate_properties.calculate_property_mols(
            [mol],
            config=create_mock_config("unknown_calculator"),
        )


# Tests for extract_cosmo_output function
def test_extract_cosmo_output_none_input():
    """Test extract_cosmo_output with None input"""
    result = calculate_properties.extract_cosmo_output(None)
    assert result is None


def test_extract_cosmo_output_success():
    """Test successful COSMO output extraction"""
    cosmo_data = "COSMO solvation energy: -5.2 kcal/mol\nArea: 235.5 Ang^2"
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        COSMO_OUTPUT_KEY: calculated_properties.StringProperty(cosmo_data),
    }

    result = calculate_properties.extract_cosmo_output(properties)

    assert result is not None
    assert result == cosmo_data


def test_extract_cosmo_output_not_present():
    """Test extract_cosmo_output when COSMO output is not in properties"""
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        "weight": calculated_properties.ScalarProperty(0.341),
    }

    result = calculate_properties.extract_cosmo_output(properties)

    assert result is None


def test_extract_cosmo_output_remove_from_dict():
    """Test extract_cosmo_output with remove_from_dict=True"""
    cosmo_data = "COSMO solvation energy: -5.2 kcal/mol"
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        COSMO_OUTPUT_KEY: calculated_properties.StringProperty(cosmo_data),
    }

    result = calculate_properties.extract_cosmo_output(properties, remove_from_dict=True)

    assert result == cosmo_data
    assert COSMO_OUTPUT_KEY not in properties
    assert "energy" in properties


def test_extract_cosmo_output_remove_from_dict_not_present():
    """Test extract_cosmo_output with remove_from_dict=True when COSMO not present"""
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        "weight": calculated_properties.ScalarProperty(0.341),
    }
    original_keys = set(properties.keys())

    result = calculate_properties.extract_cosmo_output(properties, remove_from_dict=True)

    assert result is None
    assert set(properties.keys()) == original_keys


def test_extract_cosmo_output_preserves_other_properties():
    """Test that extract_cosmo_output preserves other properties when not removing"""
    cosmo_data = "COSMO output"
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        "weight": calculated_properties.ScalarProperty(0.341),
        COSMO_OUTPUT_KEY: calculated_properties.StringProperty(cosmo_data),
    }
    original_keys = set(properties.keys())

    result = calculate_properties.extract_cosmo_output(properties)

    assert result == cosmo_data
    assert set(properties.keys()) == original_keys


# Integration tests
def test_calculate_property_mol_then_extract_cosmo():
    """Test calculating properties and then extracting COSMO output"""
    mol = get_molecule()

    mock_calculator = partial(mock_calculator_function, identifier="test", include_cosmo=True)

    with mock.patch.dict(
        calculate_properties.CALCULATOR_REGISTRY,
        {
            "test_calculator": mock_calculator,
        },
    ):
        result_list = calculate_properties.calculate_property_mol(
            mol,
            config=create_mock_config("test_calculator"),
        )

        assert result_list is not None
        props = result_list[0]
        assert props is not None

        # Extract COSMO output
        cosmo_output = calculate_properties.extract_cosmo_output(props)
        assert cosmo_output is not None
        assert "$cosmo_energy" in cosmo_output

        # Verify COSMO output was removed if we use remove_from_dict
        cosmo_output = calculate_properties.extract_cosmo_output(props, remove_from_dict=True)
        assert cosmo_output is not None
        assert COSMO_OUTPUT_KEY not in props


def test_calculator_registry_has_expected_calculators():
    """Test that CALCULATOR_REGISTRY contains expected calculators"""
    expected_calculators = [
        "xtb_single_point",
        "turbomole_single_point",
        "turbomole_freeh",
        "turbomole_fukui_indices",
        "turbomole_nmr_shielding",
        "turbomole_vcd",
        "jaguar_h_abstraction_energies",
        "jaguar_properties",
    ]

    for calculator in expected_calculators:
        assert (
            calculator in calculate_properties.CALCULATOR_REGISTRY
        ), f"Calculator {calculator} not found in CALCULATOR_REGISTRY"
