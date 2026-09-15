"""Tests for the collect CLI module."""

import shutil
from zipfile import ZipFile

import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import collect_output, compound_dir
from qm_atlas.command_line.pages import collect
from qm_atlas.command_line.pages.collect import CollectInterfaceConfig, main

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
