"""Tests for read_input.py CLI module.

Tests cover:
- Configuration loading and validation
- Single and multiple input files (SDF, CSV)
- Directory reading
- Compound directory creation
- Error handling and edge cases
"""

import shutil

import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import read_input
from qm_atlas.command_line.pages.read_input import ReadInputConfig, run, setup_logging

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
# Simple alkanes
SMILES_ETHANE = "CC"  # CHEMBL135626
SMILES_PROPANE = "CCC"  # CHEMBL135416
SMILES_BUTANE = "CCCC"  # CHEMBL134702
SMILES_PENTANE = "CCCCC"  # CHEMBL135416

# Amino acids
SMILES_GLYCINE_ZWITTERION = "[NH3+]CC(=O)[O-]"  # Zwitterionic form of glycine, CHEMBL773
SMILES_GLYCINE_NEUTRAL = "NCC(=O)O"  # CHEMBL773
SMILES_ALANINE = "NC(C)C(=O)O"  # CHEMBL279597

# Pyridine tautomers
SMILES_PYRIDINONE = "O=C1C=CC=CN1"  # CHEMBL2347636
SMILES_PYRIDINONE_TAUTOMER = "OC1=NC=CC=C1"  # tautomer of pyridinone, CHEMBL2347636

# Protonation states
SMILES_ACETIC_ACID = "CC(=O)O"
SMILES_ACETATE = "CC(=O)[O-]"
SMILES_PROPIONIC_ACID = "CCC(=O)O"

# Tautomers and Protonation states (histidine)
SMILES_HISTIDINE = "O=C([C@H](CC1=CNC=N1)N)O"  # CHEMBL17962
SMILES_HISTIDINE_TAUTOMER = "N[C@@H](Cc1cnc[nH]1)C(=O)O"  # tautomer of histidine, CHEMBL17962
SMILES_HISTIDINE_ZWITTTERION = (
    "[NH3+][C@@H](Cc1c[nH]cn1)C(=O)[O-]"  # zwitterionic form of histidine, CHEMBL17962
)
SMILES_HISTIDINE_ZWITTERION_TAUTOMER = (
    "[NH3+][C@@H](Cc1cnc[nH]1)C(=O)[O-]"  # tautomer of zwitterionic histidine, CHEMBL17962
)
SMILES_HISTIDINE_PROTONATED = (
    "[NH3+][C@@H](Cc1c[nH+]c[nH]1)C(=O)O"  # fully protonated form of histidine, CHEMBL17962
)
SMILES_HISTIDINE_PROTONATED_LESS = (
    "[NH3+][C@@H](Cc1c[nH+]c[nH]1)C(=O)[O-]"  # partially protonated form of histidine, CHEMBL17962
)
SMILES_HISTIDINE_PROTONATED_TAUTOMER = "[NH3+][C@@H](Cc1c[nH]c[nH+]1)C(=O)[O-]"  # tautomer of partially protonated histidine, CHEMBL17962

# Invalid SMILES for error testing
SMILES_INVALID_1 = "INVALID_SMILES_HERE"
SMILES_INVALID_2 = "ALSOINVALID"
SMILES_INVALID_3 = "INVALID"

# Test fixtures and constants
TEST_SDF_1 = RESOURCES / "test_geometry.sdf"
TEST_SDF_2 = RESOURCES / "chembl3586573.sdf"
TEST_CSV_FILE = RESOURCES / "test.csv"  # Will create if not exists


@pytest.fixture
def temp_work_dir():
    """Create and cleanup temporary work directory."""
    tmp_path = create_homedir_tmp_path()
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture
def temp_input_dir(temp_work_dir):
    """Create a temporary directory with test input files."""
    input_dir = temp_work_dir / "input_files"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Copy test SDF files
    if TEST_SDF_1.exists():
        shutil.copy(TEST_SDF_1, input_dir / "molecules1.sdf")
    if TEST_SDF_2.exists():
        shutil.copy(TEST_SDF_2, input_dir / "molecules2.sdf")

    return input_dir


@pytest.fixture
def temp_csv_file(temp_work_dir):
    """Create a temporary CSV file for testing."""
    csv_file = temp_work_dir / "molecules.csv"

    # Create a simple CSV with SMILES
    csv_content = f"""name,smiles
    ethane,{SMILES_ETHANE}
    propane,{SMILES_PROPANE}
    butane,{SMILES_BUTANE}
    """
    csv_file.write_text(csv_content)
    return csv_file


# ===========================================================================
# Tests for Configuration Loading
# ===========================================================================


def test_load_config_minimal(temp_work_dir):
    """Test loading configuration with minimal arguments - using config file syntax."""
    results_dir = temp_work_dir / "results"
    input_file = temp_work_dir / "test.sdf"

    # Create dummy input file
    with Chem.SDWriter(str(input_file)) as writer:
        writer.write(Chem.MolFromSmiles(SMILES_ETHANE))

    # jsonargparse handles list arguments via YAML config
    config = ReadInputConfig(
        input_file=[input_file],
        results_directory=results_dir,
    )

    assert isinstance(config, ReadInputConfig)
    assert config.input_file == [input_file]
    assert config.results_directory == results_dir
    assert isinstance(config.input_config, InputConfig)


def test_load_config_multiple_inputs(temp_work_dir):
    """Test loading configuration with multiple input files."""
    results_dir = temp_work_dir / "results"
    input_file1 = temp_work_dir / "test1.sdf"
    input_file2 = temp_work_dir / "test2.sdf"

    with Chem.SDWriter(str(input_file1)) as writer:
        writer.write(Chem.MolFromSmiles(SMILES_ETHANE))
    with Chem.SDWriter(str(input_file2)) as writer:
        writer.write(Chem.MolFromSmiles(SMILES_PROPANE))

    # Create config directly using Pydantic
    config = ReadInputConfig(
        input_file=[input_file1, input_file2],
        results_directory=results_dir,
    )

    assert len(config.input_file) == 2
    assert input_file1 in config.input_file
    assert input_file2 in config.input_file


def test_load_config_csv_options(temp_work_dir, temp_csv_file):
    """Test configuration with custom CSV options."""
    results_dir = temp_work_dir / "results"

    config = ReadInputConfig(
        input_file=[temp_csv_file],
        results_directory=results_dir,
        input_config=InputConfig(
            csv_col_name="compound_id",
            csv_col_smiles="smi",
            csv_sep=";",
        ),
    )

    assert config.input_config.csv_col_name == "compound_id"  # pylint: disable=all
    assert config.input_config.csv_col_smiles == "smi"  # pylint: disable=all
    assert config.input_config.csv_sep == ";"  # pylint: disable=all


def test_load_config_verbose(temp_work_dir):
    """Test configuration with verbose flag."""
    results_dir = temp_work_dir / "results"
    input_file = temp_work_dir / "test.sdf"

    with Chem.SDWriter(str(input_file)) as writer:
        writer.write(Chem.MolFromSmiles(SMILES_ETHANE))

    config = ReadInputConfig(
        input_file=[input_file],
        results_directory=results_dir,
        verbose=True,
    )

    assert config.verbose is True


def test_config_nonexistent_input(temp_work_dir):
    """Test that nonexistent input file raises error during validation."""
    results_dir = temp_work_dir / "results"
    input_file = temp_work_dir / "nonexistent.sdf"

    with pytest.raises(ValueError, match="Input path does not exist"):
        ReadInputConfig(
            input_file=[input_file],
            results_directory=results_dir,
        )


def test_config_unsupported_file_type(temp_work_dir):
    """Test that unsupported file type raises error during validation."""
    results_dir = temp_work_dir / "results"
    unsupported_file = temp_work_dir / "test.txt"
    unsupported_file.write_text("dummy")

    with pytest.raises(ValueError, match="Unsupported input file type"):
        ReadInputConfig(
            input_file=[unsupported_file],
            results_directory=results_dir,
        )


# ===========================================================================
# Tests for Single Input File Handling
# ===========================================================================


def test_single_sdf_file(temp_work_dir):
    """Test reading from a single SDF file."""
    results_dir = temp_work_dir / "results"
    input_file = temp_work_dir / "test.sdf"

    # Create a test SDF with 2 molecules
    with Chem.SDWriter(str(input_file)) as writer:
        mol1 = Chem.MolFromSmiles(SMILES_ETHANE)
        mol1.SetProp("_Name", "ethane")
        writer.write(mol1)

        mol2 = Chem.MolFromSmiles(SMILES_PROPANE)
        mol2.SetProp("_Name", "propane")
        writer.write(mol2)

    config = ReadInputConfig(
        input_file=[input_file],
        results_directory=results_dir,
    )

    result = run(config)

    assert len(result) == 2
    assert "ethane" in result
    assert "propane" in result
    assert (results_dir / "ethane").exists()
    assert (results_dir / "propane").exists()


def test_single_csv_file(temp_work_dir, temp_csv_file):
    """Test reading from a single CSV file."""
    results_dir = temp_work_dir / "results"

    config = ReadInputConfig(
        input_file=[temp_csv_file],
        results_directory=results_dir,
        input_config=InputConfig(),
    )

    result = run(config)

    assert len(result) >= 2  # At least ethane and propane
    assert results_dir.exists()


# ===========================================================================
# Tests for Multiple Input Files
# ===========================================================================


def test_multiple_sdf_files(temp_work_dir):
    """Test reading from multiple SDF files."""
    results_dir = temp_work_dir / "results"
    sdf_file1 = temp_work_dir / "molecules1.sdf"
    sdf_file2 = temp_work_dir / "molecules2.sdf"

    # Create first SDF file
    with Chem.SDWriter(str(sdf_file1)) as writer:
        mol1 = Chem.MolFromSmiles(SMILES_ETHANE)
        mol1.SetProp("_Name", "ethane")
        writer.write(mol1)

        mol2 = Chem.MolFromSmiles(SMILES_PROPANE)
        mol2.SetProp("_Name", "propane")
        writer.write(mol2)

    # Create second SDF file
    with Chem.SDWriter(str(sdf_file2)) as writer:
        mol3 = Chem.MolFromSmiles(SMILES_BUTANE)
        mol3.SetProp("_Name", "butane")
        writer.write(mol3)

        mol4 = Chem.MolFromSmiles(SMILES_PENTANE)
        mol4.SetProp("_Name", "pentane")
        writer.write(mol4)

    config = ReadInputConfig(
        input_file=[sdf_file1, sdf_file2],
        results_directory=results_dir,
    )

    result = run(config)

    assert len(result) == 4
    assert "ethane" in result
    assert "propane" in result
    assert "butane" in result
    assert "pentane" in result

    # Verify compound directories were created
    for mol_name in result:
        assert (results_dir / mol_name).is_dir()


def test_mixed_sdf_and_csv(temp_work_dir, temp_csv_file):
    """Test reading from both SDF and CSV files together."""
    results_dir = temp_work_dir / "results"
    sdf_file = temp_work_dir / "molecules.sdf"

    # Create SDF file with molecule
    with Chem.SDWriter(str(sdf_file)) as writer:
        mol = Chem.MolFromSmiles(SMILES_ETHANE)
        mol.SetProp("_Name", "mol_from_sdf")
        writer.write(mol)

    config = ReadInputConfig(
        input_file=[sdf_file, temp_csv_file],
        results_directory=results_dir,
        input_config=InputConfig(),
    )

    result = run(config)

    # Should have molecules from both SDF and CSV
    assert len(result) > 1
    assert "mol_from_sdf" in result
    assert results_dir.exists()


def test_directory_input(temp_input_dir, temp_work_dir):
    """Test reading from a directory containing multiple files."""
    results_dir = temp_work_dir / "results"

    # Create some molecules in the input directory
    sdf1 = temp_input_dir / "set1.sdf"
    sdf2 = temp_input_dir / "set2.sdf"
    csv1 = temp_input_dir / "data.csv"

    # Create SDF files
    with Chem.SDWriter(str(sdf1)) as writer:
        mol1 = Chem.MolFromSmiles(SMILES_ETHANE)
        mol1.SetProp("_Name", "sdf1_mol1")
        writer.write(mol1)

    with Chem.SDWriter(str(sdf2)) as writer:
        mol2 = Chem.MolFromSmiles(SMILES_PROPANE)
        mol2.SetProp("_Name", "sdf2_mol1")
        writer.write(mol2)

    # Create CSV file
    csv_content = f"""name,smiles
    csv_mol1,{SMILES_BUTANE}
    csv_mol2,{SMILES_PENTANE}
    """
    csv1.write_text(csv_content)

    config = ReadInputConfig(
        input_file=[temp_input_dir],
        results_directory=results_dir,
        input_config=InputConfig(),
    )

    result = run(config)

    # Should read all files from directory
    assert len(result) >= 4
    assert results_dir.exists()
    assert (results_dir / "sdf1_mol1").exists()
    assert (results_dir / "sdf2_mol1").exists()
    assert (results_dir / "csv_mol1").exists()
    assert (results_dir / "csv_mol2").exists()


# ===========================================================================
# Tests for Directory Creation
# ===========================================================================


def test_compound_directories_created(temp_work_dir):
    """Test that compound directories are created correctly."""
    results_dir = temp_work_dir / "results"
    sdf_file = temp_work_dir / "test.sdf"

    mol_data = [
        ("molecule_1", SMILES_ETHANE),
        ("molecule_2", SMILES_PROPANE),
        ("molecule_3", SMILES_BUTANE),
    ]

    with Chem.SDWriter(str(sdf_file)) as writer:
        for name, smiles in mol_data:
            mol = Chem.MolFromSmiles(smiles)
            mol.SetProp("_Name", name)
            writer.write(mol)

    config = ReadInputConfig(
        input_file=[sdf_file],
        results_directory=results_dir,
    )

    result = run(config)

    # Verify directories and files exist
    assert len(result) == 3
    for name, _ in mol_data:
        cpd_dir = results_dir / name
        assert cpd_dir.is_dir()
        # Check that input file was created (location depends on compound_dir structure)
        assert (cpd_dir / "input" / f"{name}.sdf").exists()


def test_results_dir_created_if_missing(temp_work_dir):
    """Test that results directory is created if it doesn't exist (when parent exists)."""
    # Create intermediate parent directory
    results_parent = temp_work_dir / "results"
    results_parent.mkdir(parents=True, exist_ok=True)

    results_dir = results_parent / "compounds"
    sdf_file = temp_work_dir / "test.sdf"

    # Parent directory exists, but results_dir doesn't
    assert not results_dir.exists()
    assert results_parent.exists()

    with Chem.SDWriter(str(sdf_file)) as writer:
        mol = Chem.MolFromSmiles(SMILES_ETHANE)
        mol.SetProp("_Name", "test_mol")
        writer.write(mol)

    config = ReadInputConfig(
        input_file=[sdf_file],
        results_directory=results_dir,
    )

    run(config)

    assert results_dir.exists()
    assert (results_dir / "test_mol").is_dir()
    assert (results_dir / "test_mol" / "input" / "test_mol.sdf").exists()


# ===========================================================================
# Tests for Error Handling
# ===========================================================================


def test_empty_sdf_file(temp_work_dir):
    """Test handling of empty SDF file."""
    results_dir = temp_work_dir / "results"
    empty_sdf = temp_work_dir / "empty.sdf"
    empty_sdf.write_text("")

    config = ReadInputConfig(
        input_file=[empty_sdf],
        results_directory=results_dir,
    )

    result = run(config)

    # Should return empty result or handle gracefully
    assert isinstance(result, dict)


def test_invalid_smiles_in_csv(temp_work_dir):
    """Test handling of invalid SMILES in CSV file."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "invalid.csv"

    csv_content = f"""name,smiles
    good_mol,{SMILES_ETHANE}
    bad_mol,{SMILES_INVALID_1}
    another_good,{SMILES_PROPANE}
    """
    csv_file.write_text(csv_content)

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(),
    )

    result = run(config)

    # Should only include valid molecules
    assert "good_mol" in result
    assert "another_good" in result
    assert "bad_mol" not in result  # Invalid SMILES should be skipped


def test_no_valid_molecules(temp_work_dir):
    """Test handling when no valid molecules are found."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "invalid.csv"

    csv_content = f"""name,smiles
    bad1,{SMILES_INVALID_3}
    bad2,{SMILES_INVALID_2}
    """
    csv_file.write_text(csv_content)

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(),
    )

    result = run(config)

    # Should handle gracefully
    assert isinstance(result, dict)
    assert len(result) == 0


# ===========================================================================
# Tests for CSV Configuration Options
# ===========================================================================


def test_custom_csv_columns(temp_work_dir):
    """Test reading CSV with custom column names."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "custom.csv"

    # Create CSV with custom column names
    csv_content = f"""molecule_id,canonical_smiles
    mol1,{SMILES_ETHANE}
    mol2,{SMILES_PROPANE}
    mol3,{SMILES_BUTANE}
    """
    csv_file.write_text(csv_content)

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(
            csv_col_name="molecule_id",
            csv_col_smiles="canonical_smiles",
        ),
    )

    result = run(config)

    assert len(result) == 3
    assert "mol1" in result
    assert "mol2" in result
    assert "mol3" in result


def test_custom_csv_separator(temp_work_dir):
    """Test reading CSV with custom separator."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "semicolon.csv"

    # Create CSV with semicolon separator
    csv_content = f"""name;smiles
    mol1;{SMILES_ETHANE}
    mol2;{SMILES_PROPANE}
    """
    csv_file.write_text(csv_content)

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(csv_sep=";"),
    )

    result = run(config)

    assert len(result) == 2
    assert "mol1" in result
    assert "mol2" in result


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_full_workflow_multiple_files(temp_work_dir):
    """Integration test: multiple files through entire workflow."""
    results_dir = temp_work_dir / "results"

    # Create multiple input files
    sdf1 = temp_work_dir / "batch1.sdf"
    sdf2 = temp_work_dir / "batch2.sdf"
    csv1 = temp_work_dir / "compounds.csv"

    # SDF batch 1
    with Chem.SDWriter(str(sdf1)) as writer:
        for i in range(2):
            mol = Chem.MolFromSmiles("C" * (i + 2))
            mol.SetProp("_Name", f"batch1_mol{i}")
            writer.write(mol)

    # SDF batch 2
    with Chem.SDWriter(str(sdf2)) as writer:
        for i in range(2):
            mol = Chem.MolFromSmiles("C" * (i + 4))
            mol.SetProp("_Name", f"batch2_mol{i}")
            writer.write(mol)

    # CSV
    csv_content = f"""name,smiles
    csv_mol1,{SMILES_ETHANE}
    csv_mol2,{SMILES_PROPANE}
    csv_mol3,{SMILES_BUTANE}
    """
    csv1.write_text(csv_content)

    # Run workflow
    config = ReadInputConfig(
        input_file=[sdf1, sdf2, csv1],
        results_directory=results_dir,
        input_config=InputConfig(),
        verbose=True,
    )

    setup_logging(verbose=config.verbose)
    result = run(config)

    # Verify results
    assert len(result) >= 7
    assert results_dir.exists()

    # Verify compound directories
    for mol_name in result:
        cpd_dir = results_dir / mol_name
        assert cpd_dir.is_dir()
        # Check that input file was created
        assert (cpd_dir / "input" / f"{mol_name}.sdf").exists()


# ===========================================================================
# Tests for explicit parent-compound grouping
# ===========================================================================


def test_csv_parent_col_groups_states(temp_work_dir):
    """Rows sharing a parent_col value should land in the same compound dir."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "states.csv"
    csv_file.write_text(
        "name,smiles,parent\n"
        f"GLY_A,{SMILES_GLYCINE_ZWITTERION},glycine\n"
        f"GLY_B,{SMILES_GLYCINE_NEUTRAL},glycine\n"
        f"ALA,{SMILES_ALANINE},alanine\n"
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(parent_col="parent"),
    )
    result = run(config)

    assert set(result.keys()) == {"glycine", "alanine"}

    glycine_dir = results_dir / "glycine"
    assert (glycine_dir / "input" / "GLY_A.sdf").is_file()
    assert (glycine_dir / "input" / "GLY_B.sdf").is_file()
    assert (results_dir / "alanine" / "input" / "ALA.sdf").is_file()

    # Both states must be registered under the glycine compound directory.
    from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir

    glycine_cpd = create_cpd_dir(glycine_dir, create_new=False)
    registered = set(glycine_cpd.registry_handler.get_registered_names())
    assert {"GLY_A", "GLY_B"}.issubset(registered)


def test_sdf_parent_property_groups_states(temp_work_dir):
    """SDF records sharing a property value should land in the same compound dir."""
    results_dir = temp_work_dir / "results"
    sdf_file = temp_work_dir / "states.sdf"

    with Chem.SDWriter(str(sdf_file)) as writer:
        m1 = Chem.MolFromSmiles(SMILES_GLYCINE_ZWITTERION)
        m1.SetProp("_Name", "GLY_A")
        m1.SetProp("parent_id", "glycine")
        writer.write(m1)

        m2 = Chem.MolFromSmiles(SMILES_GLYCINE_NEUTRAL)
        m2.SetProp("_Name", "GLY_B")
        m2.SetProp("parent_id", "glycine")
        writer.write(m2)

        m3 = Chem.MolFromSmiles(SMILES_ALANINE)
        m3.SetProp("_Name", "ALA")
        m3.SetProp("parent_id", "alanine")
        writer.write(m3)

    config = ReadInputConfig(
        input_file=[sdf_file],
        results_directory=results_dir,
        input_config=InputConfig(parent_property="parent_id"),
    )
    result = run(config)

    assert set(result.keys()) == {"glycine", "alanine"}
    assert (results_dir / "glycine" / "input" / "GLY_A.sdf").is_file()
    assert (results_dir / "glycine" / "input" / "GLY_B.sdf").is_file()


def test_sdf_parent_property_keeps_records_missing_value(temp_work_dir):
    """Records missing the parent property are kept as standalone compounds (with a warning)."""
    results_dir = temp_work_dir / "results"
    sdf_file = temp_work_dir / "partial.sdf"

    with Chem.SDWriter(str(sdf_file)) as writer:
        m1 = Chem.MolFromSmiles(SMILES_ETHANE)
        m1.SetProp("_Name", "withparent")
        m1.SetProp("parent_id", "p1")
        writer.write(m1)

        m2 = Chem.MolFromSmiles(SMILES_PROPANE)
        m2.SetProp("_Name", "noparent")
        writer.write(m2)

    config = ReadInputConfig(
        input_file=[sdf_file],
        results_directory=results_dir,
        input_config=InputConfig(parent_property="parent_id"),
    )
    result = run(config)

    # The compound with the property is grouped under its parent key.
    # The compound without the property is kept as its own standalone compound.
    assert set(result.keys()) == {"p1", "noparent"}


# ===========================================================================
# Tests for detect_states (auto tautomer / protonation-state grouping)
# ===========================================================================


def test_detect_states_groups_tautomers_from_csv(temp_work_dir):
    """Tautomers supplied via CSV should be grouped into one compound dir."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "tautomers.csv"

    # 4-hydroxypyridine and 4-pyridinone are canonical tautomers of each other.
    # Glycine neutral and zwitterion are protonation states of each other.
    csv_file.write_text(
        "name,smiles\n"
        f"pyridinone,{SMILES_PYRIDINONE}\n"  # 4(3H)-pyridinone tautomer
        f"pyridinone_taut,{SMILES_PYRIDINONE_TAUTOMER}\n"  # 4-hydroxypyridine
        f"gly_neutral,{SMILES_GLYCINE_NEUTRAL}\n"  # Glycine neutral
        f"gly_zwit,{SMILES_GLYCINE_ZWITTERION}\n"  # Glycine zwitterion
        f"unrelated,{SMILES_ETHANE}\n"  # ethane – standalone
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(detect_states=True),
    )
    result = run(config)

    # The two pyridine tautomers and two glycine states should each land in their own directory; ethane stays alone.
    assert "unrelated" in result
    assert len(result) == 3, f"Expected 3 compound groups, got {len(result)}"

    # Find the tautomer group keys (excluding ethane)
    tautomer_keys = [k for k in result if k != "unrelated"]
    assert len(tautomer_keys) == 2

    from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir

    # Check that one group contains pyridine states and one contains glycine states
    groups_registered = {}
    for key in tautomer_keys:
        cpd_dir = results_dir / key
        cpd = create_cpd_dir(cpd_dir, create_new=False)
        registered = set(cpd.registry_handler.get_registered_names())
        groups_registered[key] = registered

    # Verify one group has pyridine states
    pyridine_group = [
        k for k, v in groups_registered.items() if {"pyridinone", "pyridinone_taut"}.issubset(v)
    ]
    assert len(pyridine_group) == 1, "Pyridine states should be grouped together"

    # Verify one group has glycine states
    glycine_group = [
        k for k, v in groups_registered.items() if {"gly_neutral", "gly_zwit"}.issubset(v)
    ]
    assert len(glycine_group) == 1, "Glycine states should be grouped together"


def test_detect_states_groups_protonation_states_from_csv(temp_work_dir):
    """Different protonation states of the same compound should be grouped."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "protonation.csv"

    # Acetic acid and acetate neutralise to the same compound.
    csv_file.write_text(
        "name,smiles\n"
        f"acetic_acid,{SMILES_ACETIC_ACID}\n"
        f"acetate,{SMILES_ACETATE}\n"
        f"propane,{SMILES_PROPANE}\n"
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(detect_states=True),
    )
    result = run(config)

    assert "propane" in result
    prot_keys = [k for k in result if k != "propane"]
    assert len(prot_keys) == 1, f"Expected 1 group for protonation states, got {prot_keys}"


def test_detect_states_default_off_keeps_compounds_separate(temp_work_dir):
    """Without detect_states, tautomers stay as independent compounds."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "tautomers.csv"

    csv_file.write_text(
        "name,smiles\n"
        f"pyridinone,{SMILES_PYRIDINONE}\n"
        f"pyridinone_taut,{SMILES_PYRIDINONE_TAUTOMER}\n"
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(),  # detect_states defaults to False
    )
    result = run(config)

    assert "pyridinone" in result
    assert "pyridinone_taut" in result
    assert len(result) == 2


def test_detect_states_ignored_when_parent_col_set(temp_work_dir):
    """parent_col takes precedence; detect_states is silently ignored."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "explicit.csv"

    csv_file.write_text(
        "name,smiles,parent\n"
        f"state_a,{SMILES_ACETIC_ACID},compound1\n"
        f"state_b,{SMILES_ACETATE},compound1\n"
        f"other,{SMILES_PROPANE},compound2\n"
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(parent_col="parent", detect_states=True),
    )
    result = run(config)

    # Grouping is driven by parent_col, not auto-detection.
    assert set(result.keys()) == {"compound1", "compound2"}

    from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir

    cpd = create_cpd_dir(results_dir / "compound1", create_new=False)
    registered = set(cpd.registry_handler.get_registered_names())
    assert {"state_a", "state_b"}.issubset(registered)


def test_detect_states_ignored_when_parent_property_set(temp_work_dir):
    """parent_property takes precedence; detect_states is silently ignored."""
    results_dir = temp_work_dir / "results"
    sdf_file = temp_work_dir / "states.sdf"

    with Chem.SDWriter(str(sdf_file)) as writer:
        m1 = Chem.MolFromSmiles(SMILES_ACETIC_ACID)
        m1.SetProp("_Name", "acid")
        m1.SetProp("cpd", "comp1")
        writer.write(m1)

        # deliberately create a second state with the same parent property value
        # but which would be detected as a different state if detect_states were active
        # the parent property should override detect_states and group them together
        m2 = Chem.MolFromSmiles(SMILES_PROPIONIC_ACID)
        m2.SetProp("_Name", "anion")
        m2.SetProp("cpd", "comp1")
        writer.write(m2)

    config = ReadInputConfig(
        input_file=[sdf_file],
        results_directory=results_dir,
        input_config=InputConfig(parent_property="cpd", detect_states=True),
    )
    result = run(config)

    assert set(result.keys()) == {"comp1"}


def test_detect_states_complex_histidine_all_grouped(temp_work_dir):
    """Histidine with multiple tautomeric and protonation states should all group as one."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "histidine.csv"

    # Histidine has multiple ionizable centers (primary amine, imidazole) and tautomeric forms
    csv_file.write_text(
        "name,smiles\n"
        f"his_neutral,{SMILES_HISTIDINE}\n"
        f"his_tautomer,{SMILES_HISTIDINE_TAUTOMER}\n"
        f"his_zwit,{SMILES_HISTIDINE_ZWITTTERION}\n"
        f"his_zwit_taut,{SMILES_HISTIDINE_ZWITTERION_TAUTOMER}\n"
        f"his_protonated,{SMILES_HISTIDINE_PROTONATED}\n"
        f"his_protonated_less,{SMILES_HISTIDINE_PROTONATED_LESS}\n"
        f"his_protonated_taut,{SMILES_HISTIDINE_PROTONATED_TAUTOMER}\n"
    )

    config = ReadInputConfig(
        input_file=[csv_file],
        results_directory=results_dir,
        input_config=InputConfig(detect_states=True),
    )
    result = run(config)

    # All histidine states should be grouped as one compound
    assert len(result) == 1, f"Expected 1 compound group, got {len(result)}: {list(result.keys())}"

    from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir

    # Get the single histidine compound directory
    his_dir = results_dir / list(result.keys())[0]
    cpd = create_cpd_dir(his_dir, create_new=False)
    registered = set(cpd.registry_handler.get_registered_names())

    # All 7 histidine states should be registered under the same compound
    expected_states = {
        "his_neutral",
        "his_tautomer",
        "his_zwit",
        "his_zwit_taut",
        "his_protonated",
        "his_protonated_less",
        "his_protonated_taut",
    }
    assert expected_states.issubset(
        registered
    ), f"Expected all histidine states to be registered, got {registered}"


def test_config_roundtrip(config_roundtrip, tmp_path):
    """read_input config (top-level input_file / results_directory) round-trips."""
    results_dir = tmp_path / "results"
    _cfg, dump = config_roundtrip(
        read_input.load_config,
        {
            "input_file": [str(TEST_SDF_1.resolve())],
            "results_directory": str(results_dir),
            "verbose": True,
        },
        inject_results_dir=False,
    )
    assert dump["input_file"] == [str(TEST_SDF_1.resolve())]
    assert dump["verbose"] is True
