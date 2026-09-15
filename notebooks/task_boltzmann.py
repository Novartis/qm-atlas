# %% [markdown]
# # Task: Boltzmann Weighting and Conformer Averaging
#
# This notebook documents `qm_atlas.tasks.boltzmann`, a set of pure-Python
# helpers for turning per-conformer energies and properties into a single
# ensemble value. **No external software is required** — these functions only
# need NumPy — so this notebook runs anywhere.
#
# Typical use: you have several conformers, each with a relative energy and some
# property (a chemical shift, a descriptor, ...). You compute Boltzmann weights
# from the energies and take a weighted average of the property.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_boltzmann.py
# ```

# %% [markdown]
# ## Software configuration
#
# These helpers do not call any external program, so a software config is not
# strictly required. For consistency with the other notebooks we still check the
# `QM_ATLAS_SOFTWARE_CONFIG_FILE` variable — comment out this cell if you only
# want the pure-Python utilities.

# %%
import os
from pathlib import Path

_config_file = os.environ.get("QM_ATLAS_SOFTWARE_CONFIG_FILE")
if not _config_file:
    raise EnvironmentError(
        "QM_ATLAS_SOFTWARE_CONFIG_FILE is not set.\n"
        "Copy .env.example to .env and set the path to your software config, "
        "or export the variable before launching VS Code / Jupyter.\n"
        "See the README (Software Configuration) for details."
    )
if not Path(_config_file).is_file():
    raise FileNotFoundError(f"Software config file does not exist: {_config_file}")

print(f"Using software config: {_config_file}")

# %% [markdown]
# ## Imports and example data
#
# Say we have four conformers with the following absolute energies (in Hartree)
# and a per-conformer property (e.g. an NMR shift in ppm).

# %%
from qm_atlas.tasks import boltzmann

energies_hartree = [-284.60179, -284.60146, -284.59944, -284.59437]
shifts_ppm = [7.39, 7.51, 7.62, 8.05]

print("Supported energy units:", sorted(boltzmann.ENERGY_UNIT_TO_J_PER_MOL))

# %% [markdown]
# ## Relative energies
#
# `compute_relative_energies` shifts energies so the lowest conformer is 0 and
# converts to any unit. Reporting in kcal/mol is common.

# %%
relative_kcal = boltzmann.compute_relative_energies(
    energies_hartree, from_unit="hartree", to_unit="kcal/mol"
)
for i, rel in enumerate(relative_kcal):
    print(f"Conformer {i}: {rel:.3f} kcal/mol")

# %% [markdown]
# ## Boltzmann weights
#
# `compute_boltzmann_weights` returns normalised weights (summing to 1). Weights
# are computed relative to the lowest-energy conformer for numerical stability.

# %%
weights = boltzmann.compute_boltzmann_weights(energies_hartree, unit="hartree", temperature=298.15)
for i, w in enumerate(weights):
    print(f"Conformer {i}: weight = {w:.4f}")
print(f"Sum of weights = {sum(weights):.6f}")

# %% [markdown]
# ## Averaging a property
#
# Compare a plain (unweighted) average with the Boltzmann-weighted average of
# the property. `weighted_average` normalises the weights internally, so any
# positive weights work.

# %%
plain = boltzmann.plain_average(shifts_ppm)
weighted = boltzmann.weighted_average(shifts_ppm, weights)
print(f"Plain average shift:            {plain:.3f} ppm")
print(f"Boltzmann-weighted average:     {weighted:.3f} ppm")

# %% [markdown]
# ## Temperature dependence
#
# Higher temperatures flatten the weight distribution. Here we sweep a few
# temperatures and watch the weighted property change.

# %%
for temperature in (200.0, 298.15, 400.0, 600.0):
    w = boltzmann.compute_boltzmann_weights(energies_hartree, "hartree", temperature)
    avg = boltzmann.weighted_average(shifts_ppm, w)
    print(
        f"T = {temperature:6.1f} K -> weight of lowest-energy conformer {w[0]:.3f}, avg {avg:.3f} ppm"
    )

# %% [markdown]
# ## Unit conversion helper
#
# `energy_to_j_per_mol` converts a sequence of energies to J/mol and underpins
# the functions above. The ideal-gas constant used is
# `boltzmann.R_J_PER_MOL_K`.

# %%
import numpy as np

energies_j = boltzmann.energy_to_j_per_mol(energies_hartree, "hartree")
print("Energies (J/mol):", np.round(energies_j, 1))
print("R =", boltzmann.R_J_PER_MOL_K, "J/(mol K)")
