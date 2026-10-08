"""Tests for the collect CLI module."""

import shutil
from zipfile import ZipFile

import pandas as pd
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import collect_output, compound_dir
from qm_atlas.command_line.file_interface.compound_dir import PkaInfo
from qm_atlas.command_line.pages import collect
from qm_atlas.command_line.pages.collect import CollectInterfaceConfig, main
from qm_atlas.tasks.common import ScalarProperty

GLYCINE_RESOURCE = RESOURCES / "cpd_dir_example" / "glycine"


@pytest.fixture
def results_dir():
    """Copy the glycine compound directory into a fresh results directory."""
    tmp_path = create_homedir_tmp_path()
    results_directory = tmp_path / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GLYCINE_RESOURCE, results_directory / "glycine")
    yield results_directory
    shutil.rmtree(tmp_path, ignore_errors=True)


def _count_mols(sdf_file):
    with Chem.SDMolSupplier(str(sdf_file), removeHs=False) as supplier:
        return sum(1 for mol in supplier if mol is not None)


def test_collect_default_output_directory(results_dir):
    config = CollectInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    main(config=config)

    collected = results_dir / collect_output.OUTPUT_DIR_NAME
    assert collected.is_dir()

    merged_sdf = collected / "glycine.sdf"
    assert merged_sdf.is_file()

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    assert _count_mols(merged_sdf) == len(cpd_dir.get_result_files("glycine"))


def test_collect_copies_bookkeeping_and_cosmo(results_dir):
    config = CollectInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    main(config=config)

    collected = results_dir / collect_output.OUTPUT_DIR_NAME

    # CSV bookkeeping files that exist in the resource are copied verbatim.
    assert (collected / "glycine_properties.csv").is_file()
    assert (collected / "glycine_conformer_properties.csv").is_file()

    # COSMO files are gathered into a per-state zip archive.
    cosmo_zip = collected / "glycine_cosmo.zip"
    assert cosmo_zip.is_file()
    with ZipFile(cosmo_zip) as zf:
        names = zf.namelist()
    assert names, "COSMO archive should not be empty"
    assert all(name.endswith(".cosmo") for name in names)


def test_collect_combined_csvs(results_dir):
    config = CollectInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    main(config=config)

    collected = results_dir / collect_output.OUTPUT_DIR_NAME
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    expected_smiles = cpd_dir.registry_handler.get_entry("glycine").smiles

    molecule_df = pd.read_csv(collected / collect_output.COMBINED_MOLECULE_CSV)
    assert list(molecule_df.columns[:2]) == [
        collect_output.COMPOUND_NAME_COLUMN,
        collect_output.CANONICAL_SMILES_COLUMN,
    ]
    assert (molecule_df[collect_output.COMPOUND_NAME_COLUMN] == "glycine").all()
    assert (molecule_df[collect_output.CANONICAL_SMILES_COLUMN] == expected_smiles).all()

    conformer_df = pd.read_csv(collected / collect_output.COMBINED_CONFORMER_CSV)
    assert list(conformer_df.columns[:2]) == [
        collect_output.COMPOUND_NAME_COLUMN,
        collect_output.CANONICAL_SMILES_COLUMN,
    ]
    assert (conformer_df[collect_output.COMPOUND_NAME_COLUMN] == "glycine").all()
    assert (conformer_df[collect_output.CANONICAL_SMILES_COLUMN] == expected_smiles).all()


def test_collect_compound_summary_single_state(results_dir):
    config = CollectInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    main(config=config)

    collected = results_dir / collect_output.OUTPUT_DIR_NAME
    summary = pd.read_csv(collected / collect_output.COMPOUND_SUMMARY_CSV)
    row = summary.iloc[0]

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    assert row[collect_output.SUMMARY_COMPOUND_NAME_COLUMN] == "glycine"
    assert row[collect_output.SUMMARY_SMILES_COLUMN] == (
        cpd_dir.registry_handler.get_entry("glycine").smiles
    )
    # The directory's own state contributes unsuffixed property columns.
    assert "cosmo_logp" in summary.columns


def test_compound_summary_multistate_and_pka(tmp_path):
    cpd_path = tmp_path / "results" / "mol"
    cpd_dir = compound_dir.create_cpd_dir(cpd_path, create_new=True)

    reg = cpd_dir.registry_handler
    reg.register_state("mol", "O=C(O)c1ccccc1", 0, cpd_path / "input" / "mol.sdf")
    reg.register_state("mol_A", "O=C([O-])c1ccccc1", -1, cpd_path / "input" / "mol_A.sdf")
    reg.register_state("mol_A2", "[O-]C(=O)c1ccc([O-])cc1", -2, cpd_path / "input" / "mol_A2.sdf")
    reg.register_state("mol_BH", "O=C(O)c1cc[nH+]cc1", 1, cpd_path / "input" / "mol_BH.sdf")

    cpd_dir.add_properties_to_molecule_csv("mol", {"prop": ScalarProperty(1.0)})
    cpd_dir.add_properties_to_molecule_csv("mol_A", {"prop": ScalarProperty(2.0)})
    cpd_dir.add_properties_to_molecule_csv("mol_BH", {"prop": ScalarProperty(3.0)})

    # Two methods (ML + physics) must be ranked independently.
    cpd_dir.add_pka_info(
        PkaInfo("ACID", 9.5, "moka", parent_states=["mol"], child_states=["mol_A"])
    )
    cpd_dir.add_pka_info(
        PkaInfo("ACID", 10.0, "moka", parent_states=["mol_A"], child_states=["mol_A2"])
    )
    cpd_dir.add_pka_info(
        PkaInfo("BASE", 2.5, "moka", parent_states=["mol"], child_states=["mol_BH"])
    )
    cpd_dir.add_pka_info(
        PkaInfo("ACID", 9.0, "cosmo", parent_states=["mol"], child_states=["mol_A"])
    )
    cpd_dir.add_pka_info(
        PkaInfo("BASE", 1.5, "cosmo", parent_states=["mol"], child_states=["mol_BH"])
    )

    target = tmp_path / "collected"
    target.mkdir()
    collect_output._write_compound_summary_csv([cpd_dir], target)

    summary = pd.read_csv(target / collect_output.COMPOUND_SUMMARY_CSV)
    assert len(summary) == 1
    row = summary.iloc[0]

    assert row[collect_output.SUMMARY_COMPOUND_NAME_COLUMN] == "mol"
    assert row[collect_output.SUMMARY_SMILES_COLUMN] == "O=C(O)c1ccccc1"

    # Properties flatten into state-suffixed columns.
    assert row["prop"] == 1.0
    assert row["prop_A"] == 2.0
    assert row["prop_BH"] == 3.0

    # Each method is ranked on its own basis; acidic ascending, basic descending.
    assert row["moka acidic pKa 1"] == 9.5
    assert row["moka acidic pKa 2"] == 10.0
    assert row["moka basic pKa 1"] == 2.5
    assert row["cosmo acidic pKa 1"] == 9.0
    assert row["cosmo basic pKa 1"] == 1.5
    assert row["moka acidic pKa 1 states"] == "mol -> mol_A"
    assert row["cosmo basic pKa 1 states"] == "mol -> mol_BH"


def test_collect_custom_output_directory(results_dir):
    target = results_dir.parent / "my_collected"
    config = CollectInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        output_directory=target,
    )
    main(config=config)

    assert target.is_dir()
    assert (target / "glycine.sdf").is_file()


def test_collect_use_v2000(results_dir):
    target = results_dir.parent / "v2000_collected"
    config = CollectInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        output_directory=target,
        use_v2000=True,
    )
    main(config=config)

    content = (target / "glycine.sdf").read_text()
    # V2000 connection-table files do not carry the V3000 marker.
    assert "V3000" not in content


def test_config_roundtrip(config_roundtrip, tmp_path):
    """Collect options round-trip with a custom output dir and V2000 output."""
    out_dir = tmp_path / "collected"
    _cfg, dump = config_roundtrip(
        collect.load_config,
        {"output_directory": str(out_dir), "use_v2000": True},
    )
    assert dump["use_v2000"] is True
    assert dump["output_directory"] == str(out_dir)
