"""Tests for the qm_atlas.tasks.boltzmann module."""

import math

import numpy as np
import pytest

from qm_atlas.tasks import boltzmann


def test_energy_to_j_per_mol_units():
    # 1 Hartree in J/mol
    assert boltzmann.energy_to_j_per_mol([1.0], "hartree")[0] == pytest.approx(2625499.6394799)
    # kcal/mol and kJ/mol
    assert boltzmann.energy_to_j_per_mol([1.0], "kcal/mol")[0] == pytest.approx(4184.0)
    assert boltzmann.energy_to_j_per_mol([1.0], "kJ/mol")[0] == pytest.approx(1000.0)
    # identity for J/mol
    assert boltzmann.energy_to_j_per_mol([2.5], "J/mol")[0] == pytest.approx(2.5)


def test_energy_to_j_per_mol_is_case_insensitive():
    lower = boltzmann.energy_to_j_per_mol([1.0], "hartree")
    upper = boltzmann.energy_to_j_per_mol([1.0], "HARTREE")
    assert lower[0] == pytest.approx(upper[0])


def test_energy_to_j_per_mol_unknown_unit_raises():
    with pytest.raises(ValueError):
        boltzmann.energy_to_j_per_mol([1.0], "furlongs")


def test_compute_boltzmann_weights_normalised():
    weights = boltzmann.compute_boltzmann_weights([0.0, 0.0, 0.0], "kcal/mol", 298.15)
    assert sum(weights) == pytest.approx(1.0)
    # Degenerate energies -> equal weights
    for w in weights:
        assert w == pytest.approx(1.0 / 3.0)


def test_compute_boltzmann_weights_favours_low_energy():
    weights = boltzmann.compute_boltzmann_weights([0.0, 5.0], "kcal/mol", 298.15)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1]


def test_compute_boltzmann_weights_matches_analytic():
    # Two states, one 1 kcal/mol above the other, at 298.15 K
    temperature = 298.15
    delta_j = 1.0 * boltzmann.ENERGY_UNIT_TO_J_PER_MOL["kcal/mol"]
    expected_ratio = math.exp(-delta_j / (boltzmann.R_J_PER_MOL_K * temperature))

    weights = boltzmann.compute_boltzmann_weights([0.0, 1.0], "kcal/mol", temperature)
    assert weights[1] / weights[0] == pytest.approx(expected_ratio)


def test_compute_boltzmann_weights_invariant_to_energy_offset():
    base = boltzmann.compute_boltzmann_weights([0.0, 1.0, 2.0], "kcal/mol", 298.15)
    shifted = boltzmann.compute_boltzmann_weights([100.0, 101.0, 102.0], "kcal/mol", 298.15)
    assert base == pytest.approx(shifted)


def test_compute_boltzmann_weights_empty_raises():
    with pytest.raises(ValueError):
        boltzmann.compute_boltzmann_weights([], "hartree", 298.15)


def test_compute_boltzmann_weights_non_positive_temperature_raises():
    with pytest.raises(ValueError):
        boltzmann.compute_boltzmann_weights([0.0, 1.0], "hartree", 0.0)
    with pytest.raises(ValueError):
        boltzmann.compute_boltzmann_weights([0.0, 1.0], "hartree", -10.0)


def test_plain_average():
    assert boltzmann.plain_average([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert boltzmann.plain_average([5.0]) == pytest.approx(5.0)


def test_plain_average_empty_raises():
    with pytest.raises(ValueError):
        boltzmann.plain_average([])


def test_weighted_average_matches_numpy():
    values = [1.0, 2.0, 3.0]
    weights = [0.2, 0.3, 0.5]
    expected = float(np.average(values, weights=weights))
    assert boltzmann.weighted_average(values, weights) == pytest.approx(expected)


def test_weighted_average_normalises_weights():
    # Weights that do not sum to one should give the same result as normalised weights
    result_unnormalised = boltzmann.weighted_average([1.0, 3.0], [2.0, 2.0])
    result_normalised = boltzmann.weighted_average([1.0, 3.0], [0.5, 0.5])
    assert result_unnormalised == pytest.approx(result_normalised)
    assert result_unnormalised == pytest.approx(2.0)


def test_weighted_average_length_mismatch_raises():
    with pytest.raises(ValueError):
        boltzmann.weighted_average([1.0, 2.0], [1.0])


def test_weighted_average_zero_weights_raises():
    with pytest.raises(ValueError):
        boltzmann.weighted_average([1.0, 2.0], [0.0, 0.0])


def test_weighted_average_empty_raises():
    with pytest.raises(ValueError):
        boltzmann.weighted_average([], [])
