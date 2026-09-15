import numpy as np
import pandas as pd
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import require_software  # pylint: disable=import-error

from qm_atlas.wrappers.cosmotherm import cosmo_tasks, cpsa

COSMO_RESOURCES = RESOURCES / "cosmotherm"


@require_software("cosmotherm")
def get_cosmo_file(name: str):

    cosmo_file = COSMO_RESOURCES / f"{name}_c0.cosmo"

    if not cosmo_file.is_file():
        raise ValueError(f"File {cosmo_file} not found")

    return cosmo_file


@require_software("cosmotherm")
def test_calculate_logp(tmp_path):

    cosmo_file = get_cosmo_file("ethanol")
    reference_logp = -0.090

    calculation_result = cosmo_tasks.calculate_logp(
        [cosmo_file], scr=tmp_path, temperature_Celsius=25.0
    )
    logp = calculation_result[cosmo_tasks.COSMO_LOGP_KEY].get_property_value()

    np.testing.assert_approx_equal(reference_logp, logp, significant=1)


@require_software("cosmotherm")
def test_calculate_dg(tmp_path):
    cosmo_file = get_cosmo_file("ethanol")
    dg_reference = -4.55589  # kcal/mol

    calculation_result = cosmo_tasks.calculate_delta_g([cosmo_file], scr=tmp_path)
    dg = calculation_result[cosmo_tasks.COSMO_DELTA_G_KEY].get_property_value()

    np.testing.assert_approx_equal(dg_reference, dg, significant=2)


@require_software("cosmotherm")
def test_calculate_energies_charged(tmp_path):
    cosmo_file = get_cosmo_file("glycine_BH")

    df = cosmo_tasks.calculate_descriptors_df(
        [cosmo_file], scr=tmp_path, options=cosmo_tasks.DESCRIPTORS_SETTINGS_SIMPLIFIED
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "mu" in df.columns
    assert "E_COSMO+dE+Mu" in df.columns  # G_tot_BP
    assert "w(liq)" in df.columns  # Boltzmann weight in liquid


@require_software("cosmotherm")
def test_calculate_energies_neutral(tmp_path):

    cosmo_file = get_cosmo_file("ethanol")

    df = cosmo_tasks.calculate_descriptors_df(
        [cosmo_file], scr=tmp_path, options=cosmo_tasks.DESCRIPTORS_SETTINGS_SIMPLIFIED
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "mu" in df.columns
    assert "E_COSMO+dE+Mu" in df.columns  # G_tot_BP
    assert "w(liq)" in df.columns  # Boltzmann weight in liquid


def test_cpsa():
    cosmo_file = get_cosmo_file("amidething-confs")

    calculation_result = cpsa.extract_psa_from_cosmo_file(
        cosmo_file,
        smoothen=False,
        on_charges=True,
        lower_limit=-0.002,
        upper_limit=0.002,
    )
    psa_value = calculation_result[cpsa.COSMO_PSA_KEY].get_property_value()

    np.testing.assert_almost_equal(46.8, psa_value, decimal=1)


@require_software("cosmotherm")
@pytest.mark.parametrize(
    "neutral_mol, ionized_mol, form, expected",
    (
        ("glycine_ZH", "glycine_BH", "BASE", 1.83),
        ("glycine_ZH", "glycine_A", "ACID", 8.94),
    ),
)
def test_calculate_pka(neutral_mol: str, ionized_mol: str, form: str, expected: float, tmp_path):

    neutral_cosmo_file = get_cosmo_file(neutral_mol)
    ionized_cosmo_file = get_cosmo_file(ionized_mol)

    pka_value = cosmo_tasks.calculate_pka(
        [neutral_cosmo_file], [ionized_cosmo_file], form, scr=tmp_path
    )

    np.testing.assert_almost_equal(pka_value, expected, decimal=1)


@require_software("cosmotherm")
def test_calculate_descriptors(tmp_path):

    cosmo_file = get_cosmo_file("ethanol")

    df = cosmo_tasks.calculate_descriptors_df([cosmo_file], scr=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 39)


@require_software("cosmotherm")
def test_cosmo_perm(tmp_path):
    """Test COSMOperm calculation with ethanol conformer."""
    cosmo_file = get_cosmo_file("ethanol")

    if not cosmo_file.is_file():
        pytest.skip(f"Test file not found: {cosmo_file}")

    perm_reference = -2.13437694

    # Call the low-level function directly
    calculation_result = cosmo_tasks.calculate_cosmoperm(
        [cosmo_file], micelle_name="dmpc", temperature_Celsius=37.0, pH=7.4, scr=tmp_path
    )
    perm = calculation_result[cosmo_tasks.COSMO_PERM_KEY].get_property_value()

    np.testing.assert_approx_equal(perm_reference, perm, significant=1)
