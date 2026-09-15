"""Tests for conformer_generation module"""
from unittest import mock

import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from pydantic import BaseModel, ConfigDict
from rdkit import Chem

from qm_atlas.tasks import conformer_generation

TEST_SDF = RESOURCES / "test_geometry.sdf"


class MockOptions(BaseModel):
    """Mock options class that accepts any parameters for testing."""

    model_config = ConfigDict(extra="allow")

    backend: str


def mock_gen_config(backend: str, options: dict | None = None) -> MockOptions:
    """Build a MockOptions instance for a fake generator backend."""
    return MockOptions(backend=backend, **(options or {}))


def get_molecules() -> list[Chem.Mol]:
    """Load test molecules from SDF file"""
    mols = []
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        mols = [mol for mol in supplier]
    return mols


def get_molecule() -> Chem.Mol:
    """Load all test conformers as a single molecule"""
    mol = None
    with Chem.SDMolSupplier(str(TEST_SDF), removeHs=False) as supplier:
        for add_mol in supplier:
            mol = conformer_generation.join_conformers(mol, add_mol)
    return mol


def test_success_first_generator():
    """Test successful conformer generation with first generator"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen_1 = mock.Mock(return_value=mol_with_confs)
    mock_gen_2 = mock.Mock(return_value=None)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_conformers_with_fallback(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
        )

        assert result is not None
        assert result.GetNumConformers() == 3
        mock_gen_1.assert_called_once_with(mol)
        mock_gen_2.assert_not_called()


def test_fallback_to_second_generator():
    """Test fallback when first generator fails"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen_1 = mock.Mock(side_effect=ValueError("Test error"))
    mock_gen_2 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_conformers_with_fallback(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
        )

        assert result is not None
        assert result.GetNumConformers() == 3
        mock_gen_1.assert_called_once()
        mock_gen_2.assert_called_once()


def test_fallback_skips_empty_conformers():
    """Test that generators returning molecules with no conformers are skipped"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mol_no_confs = Chem.Mol(mol)  # Empty conformers
    mol_no_confs.RemoveAllConformers()  # Remove all conformers
    # Verify it actually has no conformers
    assert mol_no_confs.GetNumConformers() == 0
    mock_gen_1 = mock.Mock(return_value=mol_no_confs)
    mock_gen_2 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_conformers_with_fallback(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
        )

        assert result is not None
        assert result.GetNumConformers() == 3
        # Verify that both generators were called (first was skipped due to empty conformers)
        mock_gen_1.assert_called_once()
        mock_gen_2.assert_called_once()


def test_fallback_skips_none_return():
    """Test that generators returning None are skipped"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen_1 = mock.Mock(return_value=None)
    mock_gen_2 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_conformers_with_fallback(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
        )

        assert result is not None
        assert result.GetNumConformers() == 3


def test_all_generators_fail():
    """Test RuntimeError when all generators fail"""
    mol = get_molecules()[0]

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock.Mock(side_effect=ValueError("Error 1")), MockOptions),
            "test_gen_2": (mock.Mock(side_effect=RuntimeError("Error 2")), MockOptions),
        },
    ):
        with pytest.raises(RuntimeError, match="All conformer generators failed"):
            conformer_generation.generate_conformers_with_fallback(
                mol,
                conf_gens=[
                    mock_gen_config("test_gen_1"),
                    mock_gen_config("test_gen_2"),
                ],
            )


def test_kwargs_passed_correctly():
    """Test that kwargs are passed to the generator correctly"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen": (mock_gen, MockOptions),
        },
    ):
        kwargs = {"param1": "value1", "param2": 42}
        result = conformer_generation.generate_conformers_with_fallback(
            mol,
            conf_gens=[
                mock_gen_config("test_gen", kwargs),
            ],
        )

        assert result is not None
        mock_gen.assert_called_once_with(mol, **kwargs)


def test_single_generator_success():
    """Test successful conformer generation with single generator"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen": (mock_gen, MockOptions),
        },
    ):
        result = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen"),
            ],
            rms_threshold=None,  # Skip deduplication for simplicity
        )

        assert result is not None
        assert result.GetNumConformers() == 3


def test_multiple_generators_merge():
    """Test merging conformers from multiple generators"""
    mol = get_molecules()[0]
    mols = get_molecules()
    mol_confs_1 = get_molecule()

    # Create a second set of conformers from a different molecule
    mol_confs_2 = Chem.Mol(mols[1])
    mol_confs_2.RemoveAllConformers()
    for conf in mols[0].GetConformers():
        mol_confs_2.AddConformer(conf, assignId=True)
    mock_gen_1 = mock.Mock(return_value=mol_confs_1)
    mock_gen_2 = mock.Mock(return_value=mol_confs_2)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
            rms_threshold=None,  # Skip deduplication
        )

        assert result is not None
        # Both generators contribute their conformers
        assert result.GetNumConformers() == 4  # 3 + 1


def test_generator_failure_skipped():
    """Test that a failing generator is skipped and others continue"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen_1 = mock.Mock(side_effect=ValueError("Test error"))
    mock_gen_2 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        result = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
            ],
            rms_threshold=None,
        )

        assert result is not None
        assert result.GetNumConformers() == 3
        mock_gen_1.assert_called_once()
        mock_gen_2.assert_called_once()


def test_all_generators_fail_raises_error():
    """Test RuntimeError when all generators fail"""
    mol = get_molecules()[0]

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock.Mock(side_effect=ValueError("Error 1")), MockOptions),
            "test_gen_2": (mock.Mock(side_effect=RuntimeError("Error 2")), MockOptions),
        },
    ):
        with pytest.raises(RuntimeError, match="All conformer generators failed"):
            conformer_generation.generate_joint_conformer_set(
                mol,
                conf_gens=[
                    mock_gen_config("test_gen_1"),
                    mock_gen_config("test_gen_2"),
                ],
            )


def test_deduplication_with_rms_threshold():
    """Test that deduplication is called when rms_threshold is provided"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen": (mock_gen, MockOptions),
        },
    ):
        with mock.patch(
            "qm_atlas.tasks.utils.conformer_geometry.deduplicate_conformers"
        ) as mock_dedup:
            result = conformer_generation.generate_joint_conformer_set(
                mol,
                conf_gens=[
                    mock_gen_config("test_gen"),
                ],
                rms_threshold=0.3,
                rmsd_method="tradeoff",
                num_cores=2,
            )

            assert result is not None
            mock_dedup.assert_called_once()
            call_args = mock_dedup.call_args
            assert call_args.kwargs["rms_threshold"] == 0.3
            assert call_args.kwargs["method"] == "tradeoff"
            assert call_args.kwargs["num_cores"] == 2


def test_no_deduplication_when_threshold_none():
    """Test that deduplication is skipped when rms_threshold is None"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen": (mock_gen, MockOptions),
        },
    ):
        with mock.patch(
            "qm_atlas.tasks.utils.conformer_geometry.deduplicate_conformers"
        ) as mock_dedup:
            result = conformer_generation.generate_joint_conformer_set(
                mol,
                conf_gens=[
                    mock_gen_config("test_gen"),
                ],
                rms_threshold=None,
            )

            assert result is not None
            mock_dedup.assert_not_called()


def test_write_intermediates_false():
    """Test that no intermediate files are written when write_intermediates=False"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen": (mock_gen, MockOptions),
        },
    ):
        with mock.patch("hpc_funcs.files.generate_name") as mock_name:
            result = conformer_generation.generate_joint_conformer_set(
                mol,
                conf_gens=[
                    mock_gen_config("test_gen"),
                ],
                rms_threshold=None,
                write_intermediates=False,
            )

            assert result is not None
            mock_name.assert_not_called()


def test_write_intermediates_true(tmp_path):
    """Test that intermediate files are written when write_intermediates=True"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    scr_dir = tmp_path / "scratch"
    scr_dir.mkdir()
    mock_gen_1 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
        },
    ):
        mol_3d = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
            ],
            rms_threshold=None,
            trace_dir=scr_dir,
            write_intermediates=True,
        )

        assert mol_3d is not None
        assert isinstance(mol_3d, Chem.Mol)
        assert mol_3d.GetNumConformers() == 3

        # check that both intermediate and joint files were created
        sdf_files = [file for file in scr_dir.iterdir() if file.suffix == ".sdf"]
        assert len(sdf_files) == 2

        # Verify individual generator file exists
        gen_files = [f for f in sdf_files if "test_gen_1" in f.name]
        assert len(gen_files) == 1

        # Verify joint file exists
        joint_files = [f for f in sdf_files if "joint" in f.name]
        assert len(joint_files) == 1


def test_write_intermediates_with_none_and_empty_generators(tmp_path):
    """Test that generators returning None or empty molecules are skipped gracefully with write_intermediates=True"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mol_no_confs = Chem.Mol(mol)
    mol_no_confs.RemoveAllConformers()
    scr_dir = tmp_path / "scratch"
    scr_dir.mkdir()

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock.Mock(return_value=None), MockOptions),
            "test_gen_2": (mock.Mock(return_value=mol_no_confs), MockOptions),
            "test_gen_3": (mock.Mock(return_value=mol_with_confs), MockOptions),
        },
    ):
        mol_3d = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
                mock_gen_config("test_gen_3"),
            ],
            rms_threshold=None,
            trace_dir=scr_dir,
            write_intermediates=True,
        )

        assert mol_3d is not None
        assert mol_3d.GetNumConformers() == 3

        # Check that only gen_3 and joint file wrote intermediate files
        # (gen_1 returns None, gen_2 returns empty molecule, both should be skipped)
        sdf_files = [file for file in scr_dir.iterdir() if file.suffix == ".sdf"]
        assert len(sdf_files) == 2

        # Verify only test_gen_3 file exists, not test_gen_1 or test_gen_2
        gen_3_files = [f for f in sdf_files if "test_gen_3" in f.name]
        assert len(gen_3_files) == 1

        gen_1_files = [f for f in sdf_files if "test_gen_1" in f.name]
        assert len(gen_1_files) == 0

        gen_2_files = [f for f in sdf_files if "test_gen_2" in f.name]
        assert len(gen_2_files) == 0

        # Verify joint file exists
        joint_files = [f for f in sdf_files if "joint" in f.name]
        assert len(joint_files) == 1


def test_kwargs_passed_to_generators():
    """Test that kwargs are passed to each generator correctly"""
    mol = get_molecules()[0]
    mol_with_confs = get_molecule()
    mock_gen_1 = mock.Mock(return_value=mol_with_confs)
    mock_gen_2 = mock.Mock(return_value=mol_with_confs)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock_gen_1, MockOptions),
            "test_gen_2": (mock_gen_2, MockOptions),
        },
    ):
        kwargs_1 = {"param1": "value1"}
        kwargs_2 = {"param2": "value2"}

        result = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1", kwargs_1),
                mock_gen_config("test_gen_2", kwargs_2),
            ],
            rms_threshold=None,
        )

        assert result is not None
        mock_gen_1.assert_called_once_with(mol, **kwargs_1)
        mock_gen_2.assert_called_once_with(mol, **kwargs_2)


def test_partial_generator_failures_still_succeeds():
    """Test that system succeeds even if some generators fail mid-stream"""
    mol = get_molecules()[0]
    mol_with_confs_1 = get_molecule()  # 3 conformers
    mols = get_molecules()
    mol_with_confs_2 = Chem.Mol(mols[1])
    mol_with_confs_2.RemoveAllConformers()

    # Copy one conformer from first molecule to second
    mol_with_confs_2.AddConformer(list(mols[0].GetConformers())[0], assignId=True)

    with mock.patch.dict(
        conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS,
        {
            "test_gen_1": (mock.Mock(return_value=mol_with_confs_1), MockOptions),
            "test_gen_2": (mock.Mock(side_effect=Exception("Mid-stream failure")), MockOptions),
            "test_gen_3": (mock.Mock(return_value=mol_with_confs_2), MockOptions),
        },
    ):
        result = conformer_generation.generate_joint_conformer_set(
            mol,
            conf_gens=[
                mock_gen_config("test_gen_1"),
                mock_gen_config("test_gen_2"),
                mock_gen_config("test_gen_3"),
            ],
            rms_threshold=None,
        )

        assert result is not None
        # Should have conformers from gen_1 (3) and gen_3 (1) = 4
        assert result.GetNumConformers() == 4
