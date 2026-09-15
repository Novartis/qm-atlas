import context  # pylint: disable=import-error
import pytest
from context import require_software  # pylint: disable=import-error
from test_cosmo_tasks import get_cosmo_file

from qm_atlas.wrappers.cosmotherm import cosmo_cocrystal, cosmo_solubility


@require_software("cosmotherm")
@pytest.mark.parametrize(
    "molecule, params",
    (
        (
            "glycine",
            dict(temperature=25, references=None),
        ),
        (
            "glycine",
            dict(
                temperature=25,
                references=None,
                melting_temperature_C=233,
                melting_enthalpy_kJ_per_mol=21.0,
            ),
        ),
        (
            "glycine",
            dict(
                temperature=25,
                references=[
                    cosmo_solubility.ReferenceSolubility(
                        value=0.38, solvent_names=["acetone"], temperature=25
                    )
                ],
            ),
        ),
    ),
)
def test_single_solubilities(molecule: str, params: dict, tmp_path):
    # TODO : add melting point and melting enthalpy information

    cosmo_file = get_cosmo_file(molecule)
    solvents = ["1-octanol", "ethanol", "acetone"]

    res = cosmo_solubility.calculate_solubility([cosmo_file], solvents, scr=tmp_path, **params)

    assert len(res) == len(solvents)


@require_software("cosmotherm")
@pytest.mark.parametrize(
    "molecule, params",
    (
        (
            "glycine",
            dict(
                temperature=25,
                references=[
                    cosmo_solubility.ReferenceSolubility(
                        value=0.38, solvent_names=["acetone"], temperature=25
                    )
                ],
            ),
        ),
    ),
)
def test_binary_solubilities(molecule: str, params: dict, tmp_path):

    cosmo_file = get_cosmo_file(molecule)
    solvents = [["1-octanol", "ethanol"], ["acetone", "ethanol"], ["1-octanol", "acetone"]]
    n_cores = context.get_n_cores()

    sol_df = cosmo_solubility.calculate_solubility_mixture(
        [cosmo_file], solvents, scr=tmp_path, num_steps_binary=3, n_cores=n_cores, **params
    )

    assert len(sol_df) == len(solvents) * 4  # each binary .tab file has 5 rows

    parsed_sol_df = cosmo_solubility.parse_solubility_data(sol_df, num_solvents=2)

    solvent_cols = [cosmo_solubility.NEW_SOLVENT_COL.format(i) for i in range(1, 3)]
    for col_name in solvent_cols:
        assert col_name in parsed_sol_df.columns

    for solvent_1, solvent_2 in solvents:
        assert solvent_1 in parsed_sol_df[solvent_cols].values
        assert solvent_2 in parsed_sol_df[solvent_cols].values


@require_software("cosmotherm")
@pytest.mark.parametrize(
    "molecule, params",
    (
        (
            "glycine",
            dict(
                temperature=25,
            ),
        ),
    ),
)
def test_ternary_solubilities(molecule: str, params: dict, tmp_path):
    cosmo_file = get_cosmo_file(molecule)
    solvents = [["1-octanol", "ethanol", "acetone"], ["1-butanol", "2-propanol", "methanol"]]
    n_cores = context.get_n_cores()

    sol_df = cosmo_solubility.calculate_solubility_mixture(
        [cosmo_file], solvents, scr=tmp_path, num_steps_ternary=3, n_cores=n_cores, **params
    )

    assert sol_df.shape == (len(solvents) * 10, 7)  # each ternary .tab file has 10 rows

    parsed_sol_df = cosmo_solubility.parse_solubility_data(sol_df, num_solvents=3)

    solvent_cols = [cosmo_solubility.NEW_SOLVENT_COL.format(i) for i in range(1, 4)]
    for col_name in solvent_cols:
        assert col_name in parsed_sol_df.columns

    for solvent_1, solvent_2, solvent_3 in solvents:
        assert solvent_1 in parsed_sol_df[solvent_cols].values
        assert solvent_2 in parsed_sol_df[solvent_cols].values
        assert solvent_3 in parsed_sol_df[solvent_cols].values


@require_software("cosmotherm")
def test_cocrystal_screening(tmp_path):
    api_cosmo_file = get_cosmo_file("glycine")

    res_df = cosmo_cocrystal.screen_cocrystals(
        [api_cosmo_file],
        ["ethanol"],
        f_fit_cof=True,
        pZWI=True,
        scr=tmp_path,
    )

    assert res_df.shape == (1, 7)
    assert "Hex" in res_df.columns
    assert "f_fit_coformer" in res_df.columns
