"""Tests for the calculate_boltzmann_weights CLI module."""

import shutil

import pandas as pd
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.pages import calculate_boltzmann_weights
from qm_atlas.command_line.pages.calculate_boltzmann_weights import (
    BoltzmannWeightOptions,
    CalculateBoltzmannWeightsInterfaceConfig,
    main,
)

GLYCINE_RESOURCE = RESOURCES / "cpd_dir_example" / "glycine"
ENERGY_PROPERTY = "tm_sp_TotalEnergy(Ht)"


@pytest.fixture
def results_dir():
    """Copy the glycine compound directory into a fresh results directory."""
    tmp_path = create_homedir_tmp_path()
    results_directory = tmp_path / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GLYCINE_RESOURCE, results_directory / "glycine")
    yield results_directory
    shutil.rmtree(tmp_path, ignore_errors=True)


def _make_config(results_directory, **option_overrides):
    options = {
        "energy_property": ENERGY_PROPERTY,
        "energy_unit": "hartree",
        "temperature": 298.15,
        "weight_property": "boltzmann_weight",
    }
    options.update(option_overrides)
    return CalculateBoltzmannWeightsInterfaceConfig(
        input=CalculationInput(results_directory=results_directory),
        boltzmann_options=BoltzmannWeightOptions(**options),
    )


def test_boltzmann_weights_written_to_conformer_csv(results_dir):
    config = _make_config(results_dir)
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    conf_df = pd.read_csv(cpd_dir.conformer_csv)

    assert "boltzmann_weight" in conf_df.columns
    weights = conf_df["boltzmann_weight"].dropna()
    # One weight per conformer of the glycine state
    assert len(weights) == len(cpd_dir.get_result_files("glycine"))
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_boltzmann_weights_written_to_sdf(results_dir):
    config = _make_config(results_dir)
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    for sdf_file in cpd_dir.get_result_files("glycine"):
        mol = cpd_dir.extract_mol(sdf_file)
        assert mol.HasProp("boltzmann_weight")
        assert 0.0 <= mol.GetDoubleProp("boltzmann_weight") <= 1.0


def test_boltzmann_weights_custom_property_name(results_dir):
    config = _make_config(results_dir, weight_property="my_weight")
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    conf_df = pd.read_csv(cpd_dir.conformer_csv)
    assert "my_weight" in conf_df.columns


def test_boltzmann_weights_missing_energy_property_is_skipped(results_dir):
    config = _make_config(results_dir, energy_property="does_not_exist")
    # Should not raise; state is simply skipped because no energies are found.
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    conf_df = pd.read_csv(cpd_dir.conformer_csv)
    assert "boltzmann_weight" not in conf_df.columns


def test_config_roundtrip(config_roundtrip):
    """Boltzmann-weight options round-trip with non-default units / properties."""
    _cfg, dump = config_roundtrip(
        calculate_boltzmann_weights.load_config,
        {
            "boltzmann_options": {
                "energy_property": "my_energy",
                "energy_unit": "kcal/mol",
                "temperature": 310.0,
                "weight_property": "my_weight",
                "relative_energy_unit": "kJ/mol",
                "relative_energy_property": "my_rel",
            }
        },
    )
    opts = dump["boltzmann_options"]
    assert opts["energy_unit"] == "kcal/mol"
    assert opts["temperature"] == 310.0
