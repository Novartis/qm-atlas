"""
Extract charge/area to get PSA from ``*.cosmo`` files

References:
    [1]: J. Phys. Chem. A 1998, 102, 5074-5085, https://doi.org/10.1021/jp980017s

    [2]: P. Ertl, Private Communication
"""
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from qm_atlas.tasks import common as calculated_properties
from qm_atlas.units import BOHR_TO_ANGSTROM

# limits for charge densities (sigma) to consider a surface segment as polar.
# limits were determined in Ref. [2]
MLIMIT = -0.01
PLIMIT = 0.01

RAV = 0.5
RAV2 = RAV * RAV

COSMO_PSA_KEY = "cosmo_psa"


def cosmo_to_df(file: Path) -> pd.DataFrame:
    """Parse a .cosmo file's surface-segment table into a DataFrame."""
    with open(file, "r", encoding="utf-8") as cf:
        lines = cf.readlines()

    start_idx = None
    for idx, line in enumerate(lines):
        if "position (X, Y, Z)" in line:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(
            f"Could not find the start of the charge/area section in the COSMO file {file}."
        )

    rel_str = (
        "".join(lines[start_idx:])
        .replace("position (X, Y, Z)", "Position_X    Position_Y    Position_Z")
        .replace("#", "")
    )
    df = pd.read_csv(  # type: ignore
        StringIO(rel_str), sep=r"\s+", engine="python", comment="#", skip_blank_lines=True
    )
    return df


def dist_mat(coord: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix for a set of coordinates."""

    num_vecs = coord.shape[0]
    coord_squared = np.square(coord).sum(axis=1)
    X_sq_mat = coord_squared * np.ones(shape=(num_vecs, 1))
    D_squared = X_sq_mat + X_sq_mat.T - 2 * np.matmul(coord, coord.T)

    return D_squared


def average_sigma_density_and_charges(
    coord: np.ndarray,
    area: np.ndarray,
    sigma: np.ndarray,
    charge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average sigma charge density.

    Averaging is done according to eq. 11 in Reference [1]:
    J. Phys. Chem. A 1998, 102, 5074-5085, https://doi.org/10.1021/jp980017s
    """
    n = len(coord)
    dm_2 = dist_mat(coord)
    rr2 = area / np.pi
    denom_mat = ((rr2 + RAV2) * np.ones(shape=(n, 1))).T
    num_mat = ((rr2 * RAV2) * np.ones(shape=(n, 1))).T
    w = (num_mat / denom_mat) * np.exp(-dm_2 / denom_mat)
    sigma_smooth = np.matmul(sigma, w) / np.sum(w, axis=0)
    charge_smooth = np.matmul(charge, w) / np.sum(w, axis=0)
    return sigma_smooth, charge_smooth


def extract_sigma_density_and_charges(
    file: Path,
    smoothen: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (optionally smoothed) sigma density and atomic charges from a .cosmo file."""
    cosmo_df = cosmo_to_df(file)
    coord = (
        cosmo_df.loc[:, ["Position_X", "Position_Y", "Position_Z"]].to_numpy() * BOHR_TO_ANGSTROM
    )
    area = cosmo_df["area"].to_numpy()
    sigma = cosmo_df["charge/area"].to_numpy()
    charge = cosmo_df["charge"].to_numpy()
    if smoothen:
        return average_sigma_density_and_charges(coord, area, sigma, charge)
    else:
        return sigma, charge


def extract_psa(
    sigma: np.ndarray,
    charge: np.ndarray,
    area: np.ndarray,
    lower_limit: float = MLIMIT,
    upper_limit: float = PLIMIT,
    on_charges: bool = False,
) -> float:
    """Compute the polar surface area from sigma (or charge) values outside the given limits."""
    if on_charges:
        return np.sum(area[(charge < lower_limit) | (charge > upper_limit)])
    else:
        return np.sum(area[(sigma < lower_limit) | (sigma > upper_limit)])


def extract_psa_from_cosmo_file(
    file: Path,
    smoothen: bool = True,
    lower_limit: float = MLIMIT,
    upper_limit: float = PLIMIT,
    on_charges: bool = False,
) -> dict[str, calculated_properties.ScalarProperty]:
    """Extract the polar surface area (PSA) from a COSMO file.

    The smoothening of the sigma and charge values is done according to eq. 11 in Reference [1],
    the code is an extension of a previous implementation by P. Ertl (private communication).

    Args:
        file (Path):
            Path to the COSMO file.
        smoothen (bool, optional):
            Whether to smoothen the sigma and charge values. Defaults to True.
        lower_limit (float, optional):
            Lower limit for the PSA calculation. Defaults to :data:`MLIMIT`.
        upper_limit (float, optional):
            Upper limit for the PSA calculation. Defaults to :data:`PLIMIT`.
        on_charges (bool, optional):
            Whether to calculate PSA based on charges instead of sigma. Defaults to False.
            Note that the thresholds should be adapted accordingly.

    Returns:
        dict[str, calculated_properties.ScalarProperty]:
            Dictionary containing the PSA value.

    References:
        [1]: A. Klamt, V. Jonas, T. Bürger, J. C. W. Lohrenz,
        "Refinement and Parametrization of COSMO-RS",
        J. Phys. Chem. A 1998, 102, 5074-5085, https://doi.org/10.1021/jp980017s
    """
    sigma, charge = extract_sigma_density_and_charges(file, smoothen=smoothen)
    area = cosmo_to_df(file)["area"].to_numpy()
    psa_value = extract_psa(sigma, charge, area, lower_limit, upper_limit, on_charges)
    prop_dict = {COSMO_PSA_KEY: calculated_properties.ScalarProperty(psa_value)}
    return prop_dict
