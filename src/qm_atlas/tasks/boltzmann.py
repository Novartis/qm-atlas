"""Boltzmann weighting and conformer-property averaging utilities.

These helpers operate on already-calculated per-conformer scalar properties.
They are used by the ``calculate_boltzmann_weights`` and
``aggregate_conformer_properties`` command-line interfaces.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

# Ideal gas constant in J / (mol K)
R_J_PER_MOL_K = 8.314462618

# Supported energy units mapped to their conversion factor to J/mol.
EnergyUnit = Literal["hartree", "kcal/mol", "kJ/mol", "eV", "J/mol"]

ENERGY_UNIT_TO_J_PER_MOL: dict[str, float] = {
    "hartree": 2625499.6394799,  # 1 Hartree = 2625.4996 kJ/mol
    "kcal/mol": 4184.0,
    "kj/mol": 1000.0,
    "ev": 96485.3321,  # 96491.5666370759 Joule Per Mole
    "j/mol": 1.0,
}


def energy_to_j_per_mol(energies: Sequence[float], unit: str) -> np.ndarray:
    """Convert a sequence of energies to J/mol.

    Args:
        energies (Sequence[float]): Energy values in *unit*.
        unit (str): The energy unit (case-insensitive). One of the keys of
            :data:`ENERGY_UNIT_TO_J_PER_MOL`.

    Returns:
        np.ndarray: The energies converted to J/mol.

    Raises:
        ValueError: If *unit* is not a recognised energy unit.
    """
    factor = ENERGY_UNIT_TO_J_PER_MOL.get(unit.lower())
    if factor is None:
        raise ValueError(
            f"Unknown energy unit '{unit}'. "
            f"Supported units: {sorted(ENERGY_UNIT_TO_J_PER_MOL.keys())}"
        )
    return np.asarray(energies, dtype=float) * factor


def compute_boltzmann_weights(
    energies: Sequence[float],
    unit: str,
    temperature: float,
) -> list[float]:
    """Compute normalised Boltzmann weights for a set of conformer energies.

    Weights are computed relative to the lowest-energy conformer to avoid
    numerical overflow, then normalised so they sum to one.

    Args:
        energies (Sequence[float]): Per-conformer energies in *unit*.
        unit (str): Energy unit (see :func:`energy_to_j_per_mol`).
        temperature (float): Temperature in Kelvin. Must be positive.

    Returns:
        list[float]: Normalised Boltzmann weights, one per input energy.

    Raises:
        ValueError: If *energies* is empty or *temperature* is not positive.
    """
    if len(energies) == 0:
        raise ValueError("Cannot compute Boltzmann weights for an empty set of energies")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    energies_j = energy_to_j_per_mol(energies, unit)
    relative = energies_j - np.min(energies_j)
    exponentials = np.exp(-relative / (R_J_PER_MOL_K * temperature))
    weights = exponentials / exponentials.sum()
    return weights.tolist()


def compute_relative_energies(
    energies: Sequence[float],
    from_unit: str,
    to_unit: str,
) -> list[float]:
    """Compute per-conformer energy differences to the lowest-energy conformer.

    Args:
        energies (Sequence[float]): Per-conformer energies in *from_unit*.
        from_unit (str): Unit of the input energies (see
            :func:`energy_to_j_per_mol`).
        to_unit (str): Unit in which the returned differences are expressed
            (see :func:`energy_to_j_per_mol`).

    Returns:
        list[float]: Energy differences to the lowest-energy conformer,
        expressed in *to_unit*. The lowest-energy conformer has a value of 0.

    Raises:
        ValueError: If *energies* is empty or a unit is not recognised.
    """
    if len(energies) == 0:
        raise ValueError("Cannot compute relative energies for an empty set of energies")

    to_factor = ENERGY_UNIT_TO_J_PER_MOL.get(to_unit.lower())
    if to_factor is None:
        raise ValueError(
            f"Unknown energy unit '{to_unit}'. "
            f"Supported units: {sorted(ENERGY_UNIT_TO_J_PER_MOL.keys())}"
        )

    energies_j = energy_to_j_per_mol(energies, from_unit)
    relative_j = energies_j - np.min(energies_j)
    return (relative_j / to_factor).tolist()


def plain_average(
    values: Sequence[float] | np.ndarray, axis: int | None = None
) -> float | np.ndarray:
    """Return the arithmetic mean of *values*.

    Args:
        values: Per-conformer property values. May be multi-dimensional, in
            which case *axis* selects the axis to average over.
        axis (int | None): Axis to average over. ``None`` (default) averages the
            flattened input and returns a float.

    Returns:
        float | np.ndarray: The unweighted average; a float when the result is
        scalar, otherwise an array.

    Raises:
        ValueError: If *values* is empty.
    """
    arr = np.asarray(values, dtype=float)
    count = arr.shape[axis] if axis is not None else arr.size
    if count == 0:
        raise ValueError("Cannot average an empty set of values")
    result = np.mean(arr, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def weighted_average(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """Return the weighted mean of *values* using *weights*.

    Weights are normalised internally, so they need not sum to one.

    Args:
        values: Per-conformer property values. May be multi-dimensional, in
            which case *axis* selects the axis to average over.
        weights: Per-conformer weights (e.g. Boltzmann weights), one per entry
            along *axis*.
        axis (int | None): Axis to average over. ``None`` (default) averages the
            flattened input and returns a float.

    Returns:
        float | np.ndarray: The weighted average; a float when the result is
        scalar, otherwise an array.

    Raises:
        ValueError: If the inputs are empty, differ in length, or the weights
            sum to zero.
    """
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    count = values_arr.shape[axis] if axis is not None else values_arr.size
    if count == 0:
        raise ValueError("Cannot average an empty set of values")
    if weights_arr.size != count:
        raise ValueError(
            f"values and weights must have the same length, got {count} and {weights_arr.size}"
        )
    if weights_arr.sum() == 0:
        raise ValueError("Sum of weights is zero; cannot compute weighted average")

    result = np.average(values_arr, axis=axis, weights=weights_arr)
    return float(result) if np.ndim(result) == 0 else result
