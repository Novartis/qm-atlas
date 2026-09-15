import shutil

import numpy as np
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.wrappers.turbomole.single_point import read_cosmo_file

TEST_DIR = RESOURCES / "cpd_dir_example" / "glycine"


def test_get_result_files_can_exclude_reference_results():
    # Reference-optimization results live in results/ too; the resume check must be
    # able to look at conformer-expansion results only (regression: a compound whose
    # conformers were deleted but whose reference results remain must count as pending).
    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    all_results = cpd_dir.get_result_files("glycine")
    conformers_only = cpd_dir.get_result_files("glycine", include_references=False)

    assert len(conformers_only) < len(all_results)
    assert all(compound_dir.REF_NAME_ADDITION not in f.stem for f in conformers_only)
    assert any(compound_dir.REF_NAME_ADDITION in f.stem for f in all_results)


def test_cpd_dir_reading():
    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    assert cpd_dir.has_results()
    assert isinstance(str(cpd_dir), str)

    # check reference inputs exist
    ref_inputs = cpd_dir.get_reference_inputs("glycine")
    assert len(ref_inputs) > 0
    for ref_input in ref_inputs:
        assert ref_input.is_file()

    pka_log_file = cpd_dir.get_pka_log_file()
    assert isinstance(pka_log_file.name, str)

    input_sdf_files = cpd_dir.get_input_sdf_files()
    for sdf_file in input_sdf_files:
        assert sdf_file.is_file()
        log_file = cpd_dir.get_associated_log_file(sdf_file)
        assert isinstance(log_file.name, str)

        result_files = cpd_dir.get_result_files(sdf_file.stem)
        for result_sdf in result_files:
            assert result_sdf.is_file()
            assert result_sdf.suffix == ".sdf"

            cosmo_file = cpd_dir.get_associated_cosmo_file(result_sdf)
            assert cosmo_file.is_file()
            assert cosmo_file.suffix == ".cosmo"
            assert result_sdf.stem == cosmo_file.stem

            mol = cpd_dir.extract_mol(result_sdf)
            assert mol.GetNumConformers() == 1

    # Test new pKa API
    pka_info_list = cpd_dir.get_all_pka_info()
    for pka_info in pka_info_list:
        assert isinstance(pka_info, compound_dir.PkaInfo)
        assert isinstance(pka_info.pka_type, str)
        assert isinstance(pka_info.pka_value, float)
        assert isinstance(pka_info.method, str)
        assert isinstance(pka_info.parent_states, list)
        assert isinstance(pka_info.child_states, list)


def test_cpd_dir_writing():

    tmp_path = create_homedir_tmp_path()
    print(tmp_path)

    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    new_path = tmp_path / "cpd_dir_test" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)
    assert str(new_cpd_dir) == str(new_path)

    # add input structures
    input_sdf_files = cpd_dir.get_input_sdf_files()
    for sdf_file in input_sdf_files:
        input_mol = cpd_dir.extract_mol(sdf_file)
        new_sdf_file = new_cpd_dir.add_input_structure(input_mol)
        assert new_sdf_file.is_file()
        assert new_sdf_file.suffix == ".sdf"

    # add result conformers by building a multi-conformer mol from the source
    for sdf_file in input_sdf_files:
        mol_name = sdf_file.stem
        result_files = cpd_dir.get_result_files(mol_name)
        # filter out reference results to only get generated conformers
        gen_results = [f for f in result_files if compound_dir.REF_NAME_ADDITION not in f.stem]

        if not gen_results:
            continue

        # build a multi-conformer mol
        base_mol = cpd_dir.extract_mol(gen_results[0])
        base_mol.SetProp("_Name", mol_name)
        base_mol.GetConformer().SetId(0)
        for idx, res_sdf in enumerate(gen_results[1:], 1):
            conf_mol = cpd_dir.extract_mol(res_sdf)
            conf = conf_mol.GetConformer()
            conf.SetId(idx)
            base_mol.AddConformer(conf, assignId=True)

        new_result_sdfs = new_cpd_dir.add_result_conformers(base_mol)
        for res_sdf in new_result_sdfs:
            assert res_sdf.is_file()
            assert res_sdf.suffix == ".sdf"

        # add cosmo outputs for each result conformer
        for idx, res_sdf in enumerate(new_result_sdfs):
            old_cosmo = cpd_dir.get_associated_cosmo_file(gen_results[idx])
            cosmo_output = read_cosmo_file(old_cosmo.parent, old_cosmo.name, "def2-TZVPD")
            new_cpd_dir.add_cosmo_result(res_sdf, cosmo_output)
            res_cosmo = new_cpd_dir.get_associated_cosmo_file(res_sdf)
            assert res_cosmo.is_file()

    # test writing of properties
    result_files = new_cpd_dir.get_result_files("glycine")
    test_sdf = result_files[0]
    test_mol = new_cpd_dir.extract_mol(test_sdf)
    mol_properties = {
        "test_prop": calculated_properties.ScalarProperty(1.0),
        "test_prop2": calculated_properties.StringProperty("some_string_property"),
    }
    new_cpd_dir.add_properties_to_sdf(test_sdf, mol_properties)
    test_mol = new_cpd_dir.extract_mol(test_sdf)
    for prop_key in mol_properties.keys():
        assert test_mol.HasProp(prop_key)

    atom_properties = {
        "test_prop3": calculated_properties.AtomBasedProperty(
            [float(idx) for idx in range(test_mol.GetNumAtoms())]
        ),
        "test_prop4": calculated_properties.AtomBasedProperty(
            np.array([float(idx) for idx in range(test_mol.GetNumAtoms())])
        ),
        "test_prop5": calculated_properties.AtomBasedProperty({0: 1.0}),
    }

    new_cpd_dir.add_properties_to_sdf(test_sdf, atom_properties)
    test_mol = new_cpd_dir.extract_mol(test_sdf)

    for prop_key in atom_properties.keys():
        test_atom = test_mol.GetAtomWithIdx(0)
        assert test_atom.HasProp(prop_key)

    illegal_properties = {
        "illegal_prop_1": calculated_properties.AtomBasedProperty({"illegal_key": 1.0}),
    }
    with pytest.raises(ValueError):
        new_cpd_dir.add_properties_to_sdf(test_sdf, illegal_properties)

    illegal_properties = {
        "illegal_prop_2": calculated_properties.AtomBasedProperty(
            [float(idx) for idx in range(test_mol.GetNumAtoms() + 1)]
        ),
    }
    with pytest.raises(ValueError):
        new_cpd_dir.add_properties_to_sdf(test_sdf, illegal_properties)

    illegal_properties = {
        "illegal_prop_3": calculated_properties.ScalarProperty(test_mol),
    }
    with pytest.raises(ValueError):
        new_cpd_dir.add_properties_to_sdf(test_sdf, illegal_properties)

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_order():

    tmp_path = create_homedir_tmp_path()
    print(tmp_path)

    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    new_path = tmp_path / "cpd_dir_test" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)
    assert str(new_cpd_dir) == str(new_path)

    # add input structures
    input_sdf_files = cpd_dir.get_input_sdf_files()
    for sdf_file in input_sdf_files:
        input_mol = cpd_dir.extract_mol(sdf_file)
        new_sdf_file = new_cpd_dir.add_input_structure(input_mol)
        assert new_sdf_file.is_file()
        assert new_sdf_file.suffix == ".sdf"

    # add result conformers
    for sdf_file in input_sdf_files:
        mol_name = sdf_file.stem
        result_files = cpd_dir.get_result_files(mol_name)
        gen_results = [f for f in result_files if compound_dir.REF_NAME_ADDITION not in f.stem]

        if not gen_results:
            continue

        base_mol = cpd_dir.extract_mol(gen_results[0])
        base_mol.SetProp("_Name", mol_name)
        base_mol.GetConformer().SetId(0)
        for idx, res_sdf in enumerate(gen_results[1:], 1):
            conf_mol = cpd_dir.extract_mol(res_sdf)
            conf = conf_mol.GetConformer()
            conf.SetId(idx)
            base_mol.AddConformer(conf, assignId=True)

        new_result_sdfs = new_cpd_dir.add_result_conformers(base_mol)
        for res_sdf in new_result_sdfs:
            assert res_sdf.is_file()
            assert res_sdf.suffix == ".sdf"

        # add cosmo outputs
        for idx, res_sdf in enumerate(new_result_sdfs):
            old_cosmo = cpd_dir.get_associated_cosmo_file(gen_results[idx])
            cosmo_output = read_cosmo_file(old_cosmo.parent, old_cosmo.name, "def2-TZVPD")
            new_cpd_dir.add_cosmo_result(res_sdf, cosmo_output)

    # add reference structures
    for sdf_file in input_sdf_files:
        mol_name = sdf_file.stem
        new_gen_results = new_cpd_dir.get_result_files(mol_name)

        # use first generated conformer as reference input
        ref_mol = new_cpd_dir.extract_mol(new_gen_results[0])
        ref_mol.SetProp("_Name", mol_name)
        ref_mol.GetConformer().SetId(0)
        ref_input_sdf = new_cpd_dir.add_reference_input(ref_mol, conf_id=0)
        assert ref_input_sdf.is_file()
        assert ref_input_sdf.suffix == ".sdf"

        # add reference results (simulate optimized reference conformers)
        for idx, gen_sdf in enumerate(new_gen_results[:2]):
            ref_result_mol = new_cpd_dir.extract_mol(gen_sdf)
            ref_result_mol.SetProp("_Name", mol_name)
            ref_result_mol.GetConformer().SetId(idx)
            ref_result_sdf = new_cpd_dir.add_reference_result(ref_result_mol, ref_input_sdf)
            assert ref_result_sdf.is_file()
            assert ref_result_sdf.suffix == ".sdf"

        # make sure that the result files are always in the correct order:
        # generated conformers first, then reference conformers
        all_result_files = new_cpd_dir.get_result_files(mol_name)

        idcs_generated = []
        idcs_reference = []
        for idx, result_sdf in enumerate(all_result_files):
            if compound_dir.REF_NAME_ADDITION in result_sdf.stem:
                idcs_reference.append(idx)
            else:
                idcs_generated.append(idx)
        for idx_gen in idcs_generated:
            for idx_ref in idcs_reference:
                assert idx_gen < idx_ref


def test_add_pka_info():
    """Test adding a single PkaInfo entry to the compound directory."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Create and add a PkaInfo object
    pka_info = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=9.5,
        method="MOKA",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated"],
    )
    new_cpd_dir.add_pka_info(pka_info)

    # Verify the pka.csv file exists
    assert new_cpd_dir.pka_csv.is_file()

    # Retrieve and validate the stored information
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 1
    assert all_pka_info[0].pka_type == "micro"
    assert all_pka_info[0].pka_value == 9.5
    assert all_pka_info[0].method == "MOKA"
    assert all_pka_info[0].parent_states == ["glycine"]
    assert all_pka_info[0].child_states == ["glycine_deprotonated"]

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_add_multiple_pka_info():
    """Test adding multiple PkaInfo entries with different properties."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_multi" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Add first pKa entry
    pka_info1 = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=2.3,
        method="MOKA",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated1"],
    )
    new_cpd_dir.add_pka_info(pka_info1)

    # Add second pKa entry with different type
    pka_info2 = compound_dir.PkaInfo(
        pka_type="macro",
        pka_value=9.5,
        method="MOKA",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated2"],
    )
    new_cpd_dir.add_pka_info(pka_info2)

    # Add third pKa entry with different method
    pka_info3 = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=3.1,
        method="Jaguar",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated3"],
    )
    new_cpd_dir.add_pka_info(pka_info3)

    # Retrieve and validate all entries
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 3

    # Check that all entries are present and correct
    pka_values = [pka.pka_value for pka in all_pka_info]
    assert 2.3 in pka_values
    assert 9.5 in pka_values
    assert 3.1 in pka_values

    pka_methods = [pka.method for pka in all_pka_info]
    assert "MOKA" in pka_methods
    assert "Jaguar" in pka_methods

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_update_pka_info():
    """Test updating an existing PkaInfo entry (matching type, method, and states)."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_update" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Add initial pKa entry
    pka_info1 = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=9.5,
        method="MOKA",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated"],
    )
    new_cpd_dir.add_pka_info(pka_info1)

    # Verify initial state
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 1
    assert all_pka_info[0].pka_value == 9.5

    # Update the pKa value with same type, method, and states
    pka_info_updated = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=9.7,  # Updated value
        method="MOKA",
        parent_states=["glycine"],
        child_states=["glycine_deprotonated"],
    )
    new_cpd_dir.add_pka_info(pka_info_updated)

    # Verify that only one entry exists and it has the updated value
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 1
    assert all_pka_info[0].pka_value == 9.7
    assert all_pka_info[0].pka_type == "micro"
    assert all_pka_info[0].method == "MOKA"

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_pka_info_with_empty_states():
    """Test adding PkaInfo with empty parent or child states."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_empty" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Add pKa entry with empty child states
    pka_info1 = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=5.0,
        method="MOKA",
        parent_states=["glycine"],
        child_states=[],
    )
    new_cpd_dir.add_pka_info(pka_info1)

    # Add pKa entry with empty parent states
    pka_info2 = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=6.0,
        method="Jaguar",
        parent_states=[],
        child_states=["glycine_protonated"],
    )
    new_cpd_dir.add_pka_info(pka_info2)

    # Retrieve and validate
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 2

    # Check first entry
    entry1 = [p for p in all_pka_info if p.method == "MOKA"][0]
    assert entry1.parent_states == ["glycine"]
    assert entry1.child_states == []

    # Check second entry
    entry2 = [p for p in all_pka_info if p.method == "Jaguar"][0]
    assert entry2.parent_states == []
    assert entry2.child_states == ["glycine_protonated"]

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_get_all_pka_info_empty():
    """Test retrieving pKa info from a directory with no pka.csv file."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_empty_dir" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Try to get all pKa info from empty directory
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert isinstance(all_pka_info, list)
    assert len(all_pka_info) == 0

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_pka_info_with_multiple_states():
    """Test adding PkaInfo with multiple parent and child states."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_multistates" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Add pKa entry with multiple parent and child states
    pka_info = compound_dir.PkaInfo(
        pka_type="micro",
        pka_value=7.5,
        method="MOKA",
        parent_states=["glycine_state1", "glycine_state2", "glycine_state3"],
        child_states=["glycine_child1", "glycine_child2"],
    )
    new_cpd_dir.add_pka_info(pka_info)

    # Retrieve and validate
    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 1
    assert all_pka_info[0].parent_states == ["glycine_state1", "glycine_state2", "glycine_state3"]
    assert all_pka_info[0].child_states == ["glycine_child1", "glycine_child2"]

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


def test_pka_info_atom_index_roundtrip():
    """The Atom_Index column is persisted and restored, with None tolerated."""
    tmp_path = create_homedir_tmp_path()
    new_path = tmp_path / "pka_test_atom_index" / "glycine"
    new_cpd_dir = compound_dir.create_cpd_dir(new_path, create_new=True)

    # Entry that does set an atom index.
    new_cpd_dir.add_pka_info(
        compound_dir.PkaInfo(
            pka_type="BASE",
            pka_value=9.6,
            method="moka_ionic_species",
            parent_states=["glycine"],
            child_states=["glycine_BH"],
            atom_index=0,
        )
    )
    # Entry from a method that does not localise the transition.
    new_cpd_dir.add_pka_info(
        compound_dir.PkaInfo(
            pka_type="ACID",
            pka_value=4.7,
            method="cosmotherm_bp-tzvpd_macro",
            parent_states=["glycine"],
            child_states=["glycine_A"],
        )
    )

    all_pka_info = new_cpd_dir.get_all_pka_info()
    assert len(all_pka_info) == 2
    by_method = {p.method: p for p in all_pka_info}
    assert by_method["moka_ionic_species"].atom_index == 0
    assert by_method["cosmotherm_bp-tzvpd_macro"].atom_index is None

    # The CSV column must exist so downstream tools see a uniform schema.
    import pandas as pd

    df = pd.read_csv(new_cpd_dir.pka_csv)
    assert compound_dir.PKA_ATOM_INDEX_COLUMN in df.columns

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


# ---------------------------------------------------------------------------
# SMILES identity helpers
# ---------------------------------------------------------------------------


def test_canonical_smiles_helpers():
    """The isomeric and stereo-insensitive helpers behave as documented."""
    chiral = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")

    isomeric = compound_dir.canonical_smiles(chiral)
    flat = compound_dir.stereo_insensitive_smiles(chiral)

    # Isomeric keeps the stereo marker, stereo-insensitive drops it.
    assert "@" in isomeric
    assert "@" not in flat

    # Deriving the flat key from the stored isomeric SMILES matches the direct
    # stereo-insensitive canonicalisation (same code path on both sides).
    assert compound_dir.stereo_insensitive_from_smiles(isomeric) == flat

    # Unparseable SMILES yields None rather than raising.
    assert compound_dir.stereo_insensitive_from_smiles("C1CC") is None


def test_stereo_insensitive_preserves_charge():
    """Formal charge (protonation) survives the stereo-insensitive key."""
    anion = compound_dir.stereo_insensitive_smiles(Chem.MolFromSmiles("CC(N)C(=O)[O-]"))
    cation = compound_dir.stereo_insensitive_smiles(Chem.MolFromSmiles("CC([NH3+])C(=O)O"))
    neutral = compound_dir.stereo_insensitive_smiles(Chem.MolFromSmiles("CC(N)C(=O)O"))

    # Distinct protonation states must not collapse onto one another.
    assert anion != neutral
    assert cation != neutral
    assert anion != cation


# ---------------------------------------------------------------------------
# Path reverse-index: find_state_by_file
# ---------------------------------------------------------------------------


def test_find_state_by_file():
    """Every registered file (input, result, reference) resolves to its state."""
    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    # Input files.
    for sdf_file in cpd_dir.get_input_sdf_files():
        assert cpd_dir.find_state_by_file(sdf_file) == "glycine"

    # Conformer result files.
    for result_file in cpd_dir.get_result_files("glycine"):
        assert cpd_dir.find_state_by_file(result_file) == "glycine"

    # Reference inputs and their results.
    ref_inputs = cpd_dir.get_reference_inputs("glycine")
    assert len(ref_inputs) > 0
    for ref_input in ref_inputs:
        assert cpd_dir.find_state_by_file(ref_input) == "glycine"
        for ref_result in cpd_dir.get_reference_results("glycine", ref_input):
            assert cpd_dir.find_state_by_file(ref_result) == "glycine"


def test_find_state_by_file_accepts_unresolved_paths():
    """Lookup resolves both sides, so a non-normalised path still matches."""
    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)

    result_file = cpd_dir.get_result_files("glycine")[0]
    messy = result_file.parent / ".." / result_file.parent.name / result_file.name
    assert cpd_dir.find_state_by_file(messy) == "glycine"


def test_find_state_by_file_unowned_raises():
    """A file not listed in the registry raises ValueError."""
    cpd_dir = compound_dir.create_cpd_dir(TEST_DIR, create_new=False)
    with pytest.raises(ValueError):
        cpd_dir.find_state_by_file(cpd_dir.dir / "not_registered.sdf")


# ---------------------------------------------------------------------------
# Chemical matching: find_state_by_mol / find_name_by_mol
# ---------------------------------------------------------------------------


def _register_smiles_state(cpd_dir, name, smiles):
    """Register a state from a SMILES string without needing 3D structures."""
    mol = Chem.MolFromSmiles(smiles)
    cpd_dir.registry_handler.register_state(
        name=name,
        smiles=compound_dir.canonical_smiles(mol),
        charge=Chem.GetFormalCharge(mol),
        input_file_path=cpd_dir.input_dir / f"{name}.sdf",
    )


def test_find_state_by_mol_exact_match():
    """An exact isomeric SMILES match returns the registered state."""
    tmp_path = create_homedir_tmp_path()
    cpd_dir = compound_dir.create_cpd_dir(tmp_path / "chem" / "ala", create_new=True)
    _register_smiles_state(cpd_dir, "ala_S", "C[C@@H](N)C(=O)O")

    query = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
    assert cpd_dir.find_state_by_mol(query) == "ala_S"

    shutil.rmtree(tmp_path)


def test_find_state_by_mol_stereo_insensitive_fallback(caplog):
    """A molecule missing stereo falls back to a stereo-insensitive match."""
    tmp_path = create_homedir_tmp_path()
    cpd_dir = compound_dir.create_cpd_dir(tmp_path / "chem" / "ala", create_new=True)
    _register_smiles_state(cpd_dir, "ala_S", "C[C@@H](N)C(=O)O")

    # Query lacks the stereocenter (as if perceived differently from 3D).
    flat_query = Chem.MolFromSmiles("CC(N)C(=O)O")
    import logging

    with caplog.at_level(logging.WARNING):
        assert cpd_dir.find_state_by_mol(flat_query) == "ala_S"
    assert any("stereo-insensitive" in rec.message for rec in caplog.records)

    shutil.rmtree(tmp_path)


def test_find_state_by_mol_ambiguous_raises():
    """A stereo-insensitive key matching multiple states refuses to guess."""
    tmp_path = create_homedir_tmp_path()
    cpd_dir = compound_dir.create_cpd_dir(tmp_path / "chem" / "ala", create_new=True)
    _register_smiles_state(cpd_dir, "ala_S", "C[C@@H](N)C(=O)O")
    _register_smiles_state(cpd_dir, "ala_R", "C[C@H](N)C(=O)O")

    flat_query = Chem.MolFromSmiles("CC(N)C(=O)O")
    with pytest.raises(ValueError):
        cpd_dir.find_state_by_mol(flat_query)

    shutil.rmtree(tmp_path)


def test_find_state_by_mol_no_match_raises():
    """A molecule with no chemical match raises ValueError."""
    tmp_path = create_homedir_tmp_path()
    cpd_dir = compound_dir.create_cpd_dir(tmp_path / "chem" / "ala", create_new=True)
    _register_smiles_state(cpd_dir, "ala_S", "C[C@@H](N)C(=O)O")

    with pytest.raises(ValueError):
        cpd_dir.find_state_by_mol(Chem.MolFromSmiles("c1ccccc1"))

    shutil.rmtree(tmp_path)
