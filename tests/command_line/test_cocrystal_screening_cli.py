"""Tests for the cocrystal_screening CLI page.

The COSMOtherm cocrystal wrapper is monkey-patched in every test, so no
external software is required.
"""

import shutil
from unittest.mock import patch

import pandas as pd
import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
SMILES_ETHANOL = "CCO"

# On-disk name suffix of the per-compound cocrystal screening CSV
# (see compound_dir.CSV_COCRYSTAL_SCREENING_FILE).
CSV_SUFFIX = "_cocrystal_screening.csv"

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import cocrystal_screening
from qm_atlas.command_line.pages.cocrystal_screening import (
    CocrystalScreeningInterfaceConfig,
    CocrystalScreeningOptions,
    CocrystalScreeningSubmissionConfig,
    run_local,
)
from qm_atlas.command_line.pages.read_input import ReadInputConfig
from qm_atlas.command_line.pages.read_input import run as read_input_run


@pytest.fixture
def temp_work_dir():
    tmp_path = create_homedir_tmp_path()
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


def _make_compound_dir_with_cosmo(
    temp_work_dir, smiles: str = SMILES_ETHANOL, name: str = "ethanol"
):
    """Create a results dir with one compound + one state + one fake cosmo file."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "in.csv"
    csv_file.write_text(f"name,smiles\n{name},{smiles}\n")
    read_input_run(
        ReadInputConfig(
            input_file=[csv_file],
            results_directory=results_dir,
            input_config=InputConfig(),
        )
    )

    cpd_dir_path = results_dir / name
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    state_name = cpd.registry_handler.find_name_by_smiles(smiles)

    result_sdf = cpd.results_dir / f"{state_name}_c001.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    writer = Chem.SDWriter(str(result_sdf))
    writer.write(mol)
    writer.close()
    cosmo_file = result_sdf.with_suffix(".cosmo")
    cosmo_file.write_text("fake cosmo\n")
    cpd.registry_handler.add_result_files(state_name, [result_sdf])

    return results_dir, cpd_dir_path, state_name


def _build_config(cpd_dir_path, **options_kwargs) -> CocrystalScreeningInterfaceConfig:
    return CocrystalScreeningInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=CocrystalScreeningSubmissionConfig(submit=False),
        cocrystal_screening_options=CocrystalScreeningOptions(**options_kwargs),
    )


def test_cocrystal_screening_writes_csv(temp_work_dir):
    _, cpd_dir_path, state_name = _make_compound_dir_with_cosmo(temp_work_dir)

    fake_df = pd.DataFrame({"Coformer": ["urea", "nicotinamide"], "dG_cocrystal": [-1.0, -2.5]})

    with patch.object(
        cocrystal_screening.cosmo_cocrystal,
        "screen_cocrystals",
        return_value=fake_df,
    ) as mock_screen:
        run_local(
            _build_config(
                cpd_dir_path,
                coformers=["urea", "nicotinamide"],
                stoichiometry=(2, 1),
            )
        )

    mock_screen.assert_called_once()
    _, kwargs = mock_screen.call_args
    assert kwargs["coformers"] == ["urea", "nicotinamide"]
    assert kwargs["stoichiometry"] == (2, 1)

    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df["states"]) == {state_name}
    assert "Coformer" in df.columns
    assert "dG_cocrystal" in df.columns


def test_appends_across_runs(temp_work_dir):
    _, cpd_dir_path, _ = _make_compound_dir_with_cosmo(temp_work_dir)

    first_df = pd.DataFrame({"Coformer": ["urea"], "dG_cocrystal": [-1.0]})
    second_df = pd.DataFrame({"Coformer": ["caffeine"], "dG_cocrystal": [-2.0]})

    with patch.object(
        cocrystal_screening.cosmo_cocrystal,
        "screen_cocrystals",
        return_value=first_df,
    ):
        run_local(_build_config(cpd_dir_path, coformers=["urea"]))

    with patch.object(
        cocrystal_screening.cosmo_cocrystal,
        "screen_cocrystals",
        return_value=second_df,
    ):
        run_local(_build_config(cpd_dir_path, coformers=["caffeine"]))

    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert list(df["Coformer"]) == ["urea", "caffeine"]


def test_no_coformers_is_noop(temp_work_dir):
    _, cpd_dir_path, _ = _make_compound_dir_with_cosmo(temp_work_dir)

    with patch.object(
        cocrystal_screening.cosmo_cocrystal,
        "screen_cocrystals",
    ) as mock_screen:
        run_local(_build_config(cpd_dir_path))

    mock_screen.assert_not_called()
    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert not csv_path.exists()


def test_missing_cosmo_file_skips_state(temp_work_dir):
    _, cpd_dir_path, state_name = _make_compound_dir_with_cosmo(temp_work_dir)

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    for result_sdf in cpd.get_result_files(state_name):
        result_sdf.with_suffix(".cosmo").unlink()

    with patch.object(
        cocrystal_screening.cosmo_cocrystal,
        "screen_cocrystals",
    ) as mock_screen:
        run_local(_build_config(cpd_dir_path, coformers=["urea"]))

    mock_screen.assert_not_called()
    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert not csv_path.exists()


def test_config_roundtrip(config_roundtrip):
    """Cocrystal screening options (incl. tuple / bool|tuple fields) round-trip."""
    _cfg, dump = config_roundtrip(
        cocrystal_screening.load_config,
        {
            "cocrystal_screening_options": {
                "coformers": ["oxalic acid", "succinic acid"],
                "level": "bp-tzvp",
                "temperature_Celsius": 37.0,
                "stoichiometry": [2, 1],
                "f_fit": [1.0, 2.0, 3.0],
                "pZWI": True,
            }
        },
    )
    opts = dump["cocrystal_screening_options"]
    assert opts["coformers"] == ["oxalic acid", "succinic acid"]
    assert opts["stoichiometry"] == [2, 1]
    assert opts["f_fit"] == [1.0, 2.0, 3.0]
