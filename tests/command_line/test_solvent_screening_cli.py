"""Tests for the solvent_screening CLI page.

The COSMOtherm solubility wrappers are monkey-patched in every test, so no
external software is required.
"""

import shutil
from unittest.mock import patch

import pandas as pd
import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
SMILES_ETHANOL = "CCO"

# On-disk name suffix of the per-compound solubility screening CSV
# (see compound_dir.CSV_SOLUBILITY_SCREENING_FILE).
CSV_SUFFIX = "_solubility_screening.csv"

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import solvent_screening
from qm_atlas.command_line.pages.read_input import ReadInputConfig
from qm_atlas.command_line.pages.read_input import run as read_input_run
from qm_atlas.command_line.pages.solvent_screening import (
    ReferenceSolubilityConfig,
    SolventScreeningInterfaceConfig,
    SolventScreeningOptions,
    SolventScreeningSubmissionConfig,
    run_local,
)


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

    # fabricate a result sdf + paired cosmo file
    result_sdf = cpd.results_dir / f"{state_name}_c001.sdf"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
    writer = Chem.SDWriter(str(result_sdf))
    writer.write(mol)
    writer.close()
    cosmo_file = result_sdf.with_suffix(".cosmo")
    cosmo_file.write_text("fake cosmo contents\n")
    cpd.registry_handler.add_result_files(state_name, [result_sdf])

    return results_dir, cpd_dir_path, state_name


def _build_config(cpd_dir_path, **options_kwargs) -> SolventScreeningInterfaceConfig:
    return SolventScreeningInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=SolventScreeningSubmissionConfig(submit=False),
        solvent_screening_options=SolventScreeningOptions(**options_kwargs),
    )


def test_pure_solvent_screening_writes_csv(temp_work_dir):
    _, cpd_dir_path, state_name = _make_compound_dir_with_cosmo(temp_work_dir)

    fake_df = pd.DataFrame({"Solvent": ["h2o", "methanol"], "logS": [-1.2, -0.5]})

    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
        return_value=fake_df,
    ) as mock_pure, patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility_mixture",
    ) as mock_mix:
        run_local(_build_config(cpd_dir_path, solvent_names=["h2o", "methanol"]))

    mock_pure.assert_called_once()
    mock_mix.assert_not_called()

    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df["states"]) == {state_name}
    assert set(df["calculation_type"]) == {"pure"}
    assert set(df["mode"]) == {"relative"}
    assert (df["references_json"].fillna("") == "").all()
    assert "logS" in df.columns


def test_mixture_screening_writes_csv(temp_work_dir):
    _, cpd_dir_path, state_name = _make_compound_dir_with_cosmo(temp_work_dir)

    fake_df = pd.DataFrame(
        {
            "Solvent 1": ["h2o", "h2o"],
            "Solvent 2": ["methanol", "methanol"],
            "Molar Fraction 1": [1.0, 0.5],
            "Molar Fraction 2": [0.0, 0.5],
            "logS": [-1.0, -0.8],
        }
    )

    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
    ) as mock_pure, patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility_mixture",
        return_value=fake_df,
    ) as mock_mix:
        run_local(
            _build_config(
                cpd_dir_path,
                solvent_combinations=[["h2o", "methanol"]],
                num_steps_binary=1,
            )
        )

    mock_pure.assert_not_called()
    mock_mix.assert_called_once()
    _, kwargs = mock_mix.call_args
    assert kwargs["num_steps_binary"] == 1
    assert kwargs["solvent_combinations"] == [["h2o", "methanol"]]

    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df["states"]) == {state_name}
    assert set(df["calculation_type"]) == {"mixture"}


def test_appends_across_runs_and_modes(temp_work_dir):
    _, cpd_dir_path, _ = _make_compound_dir_with_cosmo(temp_work_dir)

    first_df = pd.DataFrame({"Solvent": ["h2o"], "logS": [-1.2]})
    second_df = pd.DataFrame({"Solvent": ["dmso"], "logS": [-2.1]})

    # first run: relative, no reference
    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
        return_value=first_df,
    ):
        run_local(_build_config(cpd_dir_path, solvent_names=["h2o"]))

    # second run: absolute, with a reference solubility
    ref = ReferenceSolubilityConfig(value=1.5, solvent_names=["h2o"], temperature=25.0)
    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
        return_value=second_df,
    ):
        run_local(
            _build_config(
                cpd_dir_path,
                solvent_names=["dmso"],
                references=[ref],
            )
        )

    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert list(df["mode"]) == ["relative", "absolute"]
    assert df["references_json"].iloc[0] in ("", None) or pd.isna(df["references_json"].iloc[0])
    assert "h2o" in df["references_json"].iloc[1]


def test_no_solvents_configured_is_noop(temp_work_dir):
    _, cpd_dir_path, _ = _make_compound_dir_with_cosmo(temp_work_dir)

    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
    ) as mock_pure, patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility_mixture",
    ) as mock_mix:
        run_local(_build_config(cpd_dir_path))

    mock_pure.assert_not_called()
    mock_mix.assert_not_called()
    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert not csv_path.exists()


def test_missing_cosmo_file_skips_state(temp_work_dir):
    _, cpd_dir_path, state_name = _make_compound_dir_with_cosmo(temp_work_dir)

    # delete the cosmo file
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    for result_sdf in cpd.get_result_files(state_name):
        result_sdf.with_suffix(".cosmo").unlink()

    with patch.object(
        solvent_screening.cosmo_solubility,
        "calculate_solubility",
    ) as mock_pure:
        run_local(_build_config(cpd_dir_path, solvent_names=["h2o"]))

    mock_pure.assert_not_called()
    csv_path = cpd_dir_path / f"{cpd_dir_path.name}{CSV_SUFFIX}"
    assert not csv_path.exists()


def test_config_roundtrip(config_roundtrip):
    """Pure + mixture solvent screening options, incl. nested reference configs."""
    _cfg, dump = config_roundtrip(
        solvent_screening.load_config,
        {
            "solvent_screening_options": {
                "solvent_names": ["water", "methanol"],
                "solvent_combinations": [["water", "methanol"]],
                "level": "bp-tzvp",
                "melting_temperature_C": 150.0,
                "melting_enthalpy_kJ_per_mol": 25.0,
                "num_steps_binary": 11,
                "references": [
                    {"value": 0.12, "solvent_names": ["water"], "temperature": 25.0},
                    {
                        "value": 0.05,
                        "solvent_names": ["water", "methanol"],
                        "temperature": 25.0,
                        "mass_fractions": [0.5, 0.5],
                    },
                ],
            }
        },
    )
    opts = dump["solvent_screening_options"]
    assert opts["solvent_combinations"] == [["water", "methanol"]]
    assert [r["value"] for r in opts["references"]] == [0.12, 0.05]
