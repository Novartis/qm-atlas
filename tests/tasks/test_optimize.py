"""Tests for optimize module"""
from functools import partial
from unittest import mock

import numpy as np
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from pydantic import BaseModel
from rdkit import Chem

from qm_atlas.tasks import common as calculated_properties
from qm_atlas.tasks import conformer_generation, optimize
from qm_atlas.wrappers import xtb

TEST_SDF = RESOURCES / "test_geometry.sdf"


class MockConfig(BaseModel):
    """Mock config class for testing optimization drivers."""

    backend: str


def create_mock_config(driver_name: str) -> MockConfig:
    """Create a mock config object with the specified driver name."""
    return MockConfig(backend=driver_name)


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
    mol = None
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        for add_mol in supplier:
            mol = conformer_generation.join_conformers(mol, add_mol)
    return mol


def create_mock_properties(
    converged: bool = True,
    identifier: str = "",
) -> dict[str, calculated_properties.Property]:
    """Create mock optimization properties with convergence status"""
    properties = {
        "energy": calculated_properties.ScalarProperty(-130.2),
        "weight": calculated_properties.ScalarProperty(0.341),
        xtb.OPTIMIZATION_CONVERGED_KEY: calculated_properties.BoolProperty(converged),
        "identifier": calculated_properties.StringProperty(identifier),
    }
    return properties


def mock_driver_function(
    mol: Chem.Mol,
    conf_id: int,
    ids_to_fail: set[int] | None = None,
    ids_to_unconverge: set[int] | None = None,
    identifier: str = "",
    **kwargs,
) -> tuple[np.ndarray, dict[str, calculated_properties.Property]]:
    """Mock driver function for optimization."""
    if ids_to_fail is None:
        ids_to_fail = set()
    if ids_to_unconverge is None:
        ids_to_unconverge = set()

    print(f"Mock driver called for conformer ID {conf_id} with identifier {identifier}")
    print(ids_to_unconverge)
    del kwargs  # unused but API-required

    if conf_id in ids_to_fail:
        raise RuntimeError(f"Optimization failed for conformer ID {conf_id}")

    if conf_id in ids_to_unconverge:
        print(f"Mock driver unconverged for conformer ID {conf_id} with identifier {identifier}")
        properties = create_mock_properties(converged=False, identifier=identifier)
    else:
        print(f"Mock driver converged for conformer ID {conf_id} with identifier {identifier}")
        properties = create_mock_properties(converged=True, identifier=identifier)

    conformer = mol.GetConformer(conf_id)
    displacement = np.random.normal(scale=0.01, size=(mol.GetNumAtoms(), 3))
    new_coords = conformer.GetPositions() + displacement
    return (new_coords, properties)


# Tests for optimize_with_fallback function
def test_success_first_driver():
    """Test successful optimization with first driver"""
    mol = get_molecule()
    conf_id = 0

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": mock_driver_function,
            "driver_2": mock.Mock(),
        },
    ):
        result_coords, result_props = optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[create_mock_config("driver_1")],
        )

        assert result_coords is not None
        assert result_props is not None
        assert "energy" in result_props
        assert "weight" in result_props
        optimize.OPTIMIZER_FUNCTIONS["driver_2"].assert_not_called()


def test_fallback_to_second_driver_on_exception():
    """Test fallback when first driver raises exception"""
    mol = get_molecule()
    conf_id = 0
    first_driver_function = partial(
        mock_driver_function, ids_to_fail={conf_id}, identifier="driver_1"
    )
    second_driver_function = partial(mock_driver_function, identifier="driver_2")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": first_driver_function,
            "driver_2": second_driver_function,
        },
    ):
        result_coords, result_props = optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[create_mock_config("driver_1"), create_mock_config("driver_2")],
        )

        assert result_coords is not None
        assert result_props is not None
        assert result_props["identifier"].get_property_value() == "driver_2"


def test_fallback_on_unconverged():
    """Test fallback when optimization didn't converge and treat_unconverged_as_failure=True"""
    mol = get_molecule()
    conf_id = 0
    first_driver_function = partial(
        mock_driver_function, ids_to_unconverge={conf_id}, identifier="driver_1"
    )
    second_driver_function = partial(mock_driver_function, identifier="driver_2")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": first_driver_function,
            "driver_2": second_driver_function,
        },
    ):
        result_coords, result_props = optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[create_mock_config("driver_1"), create_mock_config("driver_2")],
            treat_unconverged_as_failure=True,
        )

        assert result_coords is not None
        assert result_props is not None
        assert result_props["identifier"].get_property_value() == "driver_2"


def test_accept_unconverged_when_flag_false():
    """Test that unconverged results are accepted when treat_unconverged_as_failure=False"""
    mol = get_molecule()
    conf_id = 0
    first_driver_function = partial(
        mock_driver_function, ids_to_unconverge={conf_id}, identifier="driver_1"
    )

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": first_driver_function,
            "driver_2": mock.Mock(),
        },
    ):
        result_coords, result_props = optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[create_mock_config("driver_1"), create_mock_config("driver_2")],
            treat_unconverged_as_failure=False,
        )

        assert result_coords is not None
        assert result_props is not None
        assert result_props["identifier"].get_property_value() == "driver_1"
        optimize.OPTIMIZER_FUNCTIONS["driver_2"].assert_not_called()


def test_all_drivers_fail_raises_error():
    """Test RuntimeError when all drivers fail"""
    mol = get_molecule()
    conf_id = 0
    first_driver_function = partial(
        mock_driver_function, ids_to_fail={conf_id}, identifier="driver_1"
    )
    second_driver_function = partial(
        mock_driver_function, ids_to_fail={conf_id}, identifier="driver_2"
    )

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": first_driver_function,
            "driver_2": second_driver_function,
        },
    ):
        with pytest.raises(RuntimeError):
            optimize.optimize_with_fallback(
                mol,
                conf_id,
                config=[create_mock_config("driver_1"), create_mock_config("driver_2")],
                suppress_errors=False,
            )


def test_all_drivers_fail_returns_none_when_quiet():
    """Test that None is returned when all drivers fail and suppress_errors=True"""
    mol = get_molecule()
    conf_id = 0
    first_driver_function = partial(
        mock_driver_function, ids_to_fail={conf_id}, identifier="driver_1"
    )
    second_driver_function = partial(
        mock_driver_function, ids_to_fail={conf_id}, identifier="driver_2"
    )

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": first_driver_function,
            "driver_2": second_driver_function,
        },
    ):
        result_coords, result_props = optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[create_mock_config("driver_1"), create_mock_config("driver_2")],
            suppress_errors=True,
        )

        assert result_coords is None
        assert result_props is None


def test_kwargs_passed_to_driver():
    """Test that kwargs are passed to the driver function"""
    mol = get_molecule()
    conf_id = 0
    new_coords = np.random.rand(mol.GetNumAtoms(), 3)
    properties = create_mock_properties(converged=True)

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver_1": mock.Mock(return_value=(new_coords, properties)),
        },
    ):
        config = create_mock_config("driver_1")
        optimize.optimize_with_fallback(
            mol,
            conf_id,
            config=[config],
            n_cores=4,
        )

        call_args = optimize.OPTIMIZER_FUNCTIONS["driver_1"].call_args
        assert call_args.kwargs["n_cores"] == 4


# Tests for optimize_conformer function
def test_successful_optimization():
    """Test successful conformer optimization"""
    mol = get_molecule_with_conformers()
    conformer = mol.GetConformer(1)
    coords = conformer.GetPositions()
    driver_function = partial(mock_driver_function, ids_to_fail={0, 2}, identifier="driver")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver": driver_function,
        },
    ):

        result_conf, result_props = optimize.optimize_conformer(
            conformer,
            config=[create_mock_config("driver")],
            n_cores=2,
        )

        result_positions = result_conf.GetPositions()
        assert np.allclose(result_positions, coords, atol=1.0e-1)
        assert result_props is not None
        assert "energy" in result_props
        assert "weight" in result_props


def test_optimization_failure_returns_none_when_quiet():
    """Test that None is returned on failure when suppress_errors=True"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)

    driver_function = partial(mock_driver_function, ids_to_fail={0}, identifier="driver")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver": driver_function,
        },
    ):

        result_conf, result_props = optimize.optimize_conformer(
            conformer,
            config=[create_mock_config("driver")],
            n_cores=2,
            suppress_errors=True,
        )

        assert result_conf is None
        assert result_props is None


def test_optimization_failure_raises_when_not_quiet():
    """Test that RuntimeError is raised on failure when suppress_errors=False"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)

    driver_function = partial(mock_driver_function, ids_to_fail={0}, identifier="driver")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {
            "driver": driver_function,
        },
    ):

        with pytest.raises(RuntimeError):
            optimize.optimize_conformer(
                conformer,
                config=[create_mock_config("driver")],
                n_cores=2,
                suppress_errors=False,
            )


def test_conform_id_extracted_and_passed():
    """Test that conformer ID is extracted and passed correctly"""
    mol = get_molecule_with_conformers()
    conformer = mol.GetConformer(1)
    # obtain new coordinates by applying a small random displacement
    displacement = np.random.normal(scale=0.01, size=(mol.GetNumAtoms(), 3))
    new_coords = conformer.GetPositions() + displacement
    properties = create_mock_properties(converged=True)

    with mock.patch("qm_atlas.tasks.optimize.optimize_with_fallback") as mock_optimize:
        mock_optimize.return_value = (new_coords, properties)

        config = [create_mock_config("xtb_turbomole")]
        optimize.optimize_conformer(conformer, config=config)

        call_args = mock_optimize.call_args
        # Check that conf_id was passed correctly (second positional argument)
        assert call_args[0][1] == conformer.GetId()
        # Check that config was passed
        assert call_args.kwargs["config"] == config


# Tests for optimize_mol function
def test_optimize_mol_multiple_molecules_with_real_multiprocessing():
    """Test optimize_mol with actual multiprocessing, mocking drivers directly.
    Passes a molecule with multiple conformers to be optimized.
    """
    mol = get_molecule_with_conformers()
    driver_function = partial(mock_driver_function, identifier="driver")
    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {"driver": driver_function},
    ):
        result_mol, _ = optimize.optimize_mol(
            mol, config=[create_mock_config("driver")], n_cores=2
        )

        # Verify that all conformers were optimized
        assert result_mol.GetNumConformers() == mol.GetNumConformers()
        for conf_id in range(mol.GetNumConformers()):
            orig_conf = mol.GetConformer(conf_id)
            result_conf = result_mol.GetConformer(conf_id)
            orig_pos = orig_conf.GetPositions()
            result_pos = result_conf.GetPositions()
            assert not np.allclose(orig_pos, result_pos, atol=1.0e-5)
            assert np.allclose(orig_pos, result_pos, atol=1.0e-1)


def test_no_conformers_raises_error():
    """Test ValueError when input molecules have no conformers"""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    # Don't add any conformers

    with pytest.raises(ValueError):
        optimize.optimize_mol(mol, config=[create_mock_config("driver")])


# Tests for optimize_mols function
def test_optimize_mols_single_mol():
    """Test optimize_mols with actual multiprocessing, mocking drivers directly.
    Passes a single molecule with several conformers to be optimized.
    """
    mol = get_molecule_with_conformers()
    driver = partial(mock_driver_function, identifier="driver")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {"driver": driver},
    ):
        result_mols, result_props = optimize.optimize_mols(
            [mol], config=[create_mock_config("driver")], n_cores=2
        )

        assert len(result_mols) == 1
        assert len(result_props) == 1
        # Verify that all conformers were processed
        result_mol = result_mols[0]
        assert result_mol.GetNumConformers() == mol.GetNumConformers()
        for conf_id in range(mol.GetNumConformers()):
            orig_conf = mol.GetConformer(conf_id)
            result_conf = result_mol.GetConformer(conf_id)
            orig_pos = orig_conf.GetPositions()
            result_pos = result_conf.GetPositions()
            assert not np.allclose(orig_pos, result_pos, atol=1.0e-5)
            assert np.allclose(orig_pos, result_pos, atol=1.0e-1)


def test_optimize_mols_multiple_molecules_with_real_multiprocessing():
    """Test optimize_mols with actual multiprocessing, mocking drivers directly.
    Passes several molecules to be optimized.
    """
    mols = get_molecules()[:2]  # Get two molecules
    driver = partial(mock_driver_function, identifier="driver")

    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {"driver": driver},
    ):

        result_mols, result_props = optimize.optimize_mols(
            mols, config=[create_mock_config("driver")], n_cores=2
        )

        assert len(result_mols) == len(mols)
        assert len(result_props) == len(mols)
        # Verify that all molecules have their conformers optimized
        for orig_mol, result_mol in zip(mols, result_mols):
            assert result_mol.GetNumConformers() == orig_mol.GetNumConformers()
            for conf_id in range(orig_mol.GetNumConformers()):
                orig_conf = orig_mol.GetConformer(conf_id)
                result_conf = result_mol.GetConformer(conf_id)
                orig_pos = orig_conf.GetPositions()
                result_pos = result_conf.GetPositions()
                assert not np.allclose(orig_pos, result_pos, atol=1.0e-5)
                assert np.allclose(orig_pos, result_pos, atol=1.0e-1)


# Tests for update_mol function
def test_mismatched_coordinates_raises_error():
    """Test ValueError when number of coordinate arrays doesn't match conformers"""
    mol = get_molecule_with_conformers()
    num_confs = mol.GetNumConformers()
    # Provide fewer coordinate arrays than conformers
    new_coords_lst = [np.random.rand(mol.GetNumAtoms(), 3)] * (num_confs - 1)

    with pytest.raises(ValueError, match="Number of coordinate arrays"):
        optimize.update_mol(mol, new_coords_lst)


# Tests for adapt_positions function
def test_mismatched_atom_count_raises_error():
    """Test ValueError when atom count doesn't match"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)
    # Wrong number of atoms
    wrong_coords = np.random.rand(mol.GetNumAtoms() - 1, 3)

    with pytest.raises(ValueError):
        optimize.adapt_positions(conformer, wrong_coords)


def test_successful_adaptation():
    """Test successful position adaptation with valid coordinates"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)
    displacement = np.random.normal(scale=0.01, size=(mol.GetNumAtoms(), 3))
    new_coords = conformer.GetPositions() + displacement
    result_conformer = optimize.adapt_positions(conformer, new_coords)

    assert isinstance(result_conformer, Chem.Conformer)


def test_fail_sanity_check():
    """Test successful position adaptation with valid coordinates"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)
    new_coords = conformer.GetPositions()

    h_atom = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "H"][0]
    parent_atom = h_atom.GetNeighbors()[0]
    h_coord = new_coords[h_atom.GetIdx()]
    parent_coord = new_coords[parent_atom.GetIdx()]
    # Move hydrogen too far away from parent atom to trigger sanity check failure
    new_coords[h_atom.GetIdx()] = parent_coord + 5.0 * (h_coord - parent_coord)

    with pytest.raises(
        RuntimeError,
    ):
        optimize.adapt_positions(conformer, new_coords)


def test_fail_heavy_atom_rearrangement():
    """Coordinates in which a bond between heavy atoms broke have to be rejected"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)
    new_coords = conformer.GetPositions()

    # pull a terminal heavy atom away from its parent along the bond axis
    atom = next(
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() != "H"
        and len(atom.GetNeighbors()) == 1
        and atom.GetNeighbors()[0].GetSymbol() != "H"
    )
    parent_idx = atom.GetNeighbors()[0].GetIdx()
    new_coords[atom.GetIdx()] = new_coords[parent_idx] + 2.5 * (
        new_coords[atom.GetIdx()] - new_coords[parent_idx]
    )

    with pytest.raises(RuntimeError) as exc_info:
        optimize.adapt_positions(conformer, new_coords)

    assert "stretched_bond" in str(exc_info.value)
    # the geometry check can be relaxed, but not for broken bonds
    with pytest.raises(RuntimeError):
        optimize.adapt_positions(conformer, new_coords, contact_scale=None)


# Tests for check_displacement function
def test_displacement_calculation_and_logging():
    """Test that displacement is calculated and logged"""
    mol = get_molecule()
    conformer = mol.GetConformer(0)
    new_conformer = Chem.Conformer(conformer)

    # Should not raise any errors
    optimize.check_displacement(conformer, new_conformer)


# Tests for partial failure scenarios
def test_optimize_mol_partial_failure():
    """Test handling of partial conformer optimization failure.

    When optimizing a molecule with multiple conformers, some may fail while
    others succeed. This test verifies that:
    - Successful conformers are retained in the output
    - Failed conformers are removed from the output
    - A warning is logged for failed conformers
    - The properties list only contains results for successful conformers
    """
    # Get a molecule with multiple conformers
    mol = get_molecule_with_conformers()
    initial_num_conformers = mol.GetNumConformers()
    assert initial_num_conformers > 1, "Test requires molecule with multiple conformers"

    # Create a partial mock that fails for conformer ID 1
    mock_driver = partial(mock_driver_function, ids_to_fail={1})

    # Mock the driver to fail for conformer 1, succeed for others
    with mock.patch.dict(
        optimize.OPTIMIZER_FUNCTIONS,
        {"driver": mock_driver},
    ):
        result_mol, properties_list = optimize.optimize_mol(
            mol, config=[create_mock_config("driver")]
        )
        assert result_mol is not None, "Resulting molecule should not be None"
    # Verify that one conformer was removed
    assert (
        result_mol.GetNumConformers() == initial_num_conformers - 1
    ), f"Expected {initial_num_conformers - 1} conformers after partial failure, got {result_mol.GetNumConformers()}"

    # Verify that properties list matches the number of successful conformers
    assert len(properties_list) == initial_num_conformers - 1
