# %% [markdown]
# # Task: Calculate Molecular Properties
#
# This notebook documents `qm_atlas.tasks.calculate_properties`, which runs
# single-point property calculations on the conformers of a molecule. Each
# calculator has its own options class; the `calculator_name` field selects the
# backend automatically.
#
# | Options class | Property/backend | External program |
# | --- | --- | --- |
# | `XtbSinglePointOptions` | energy, (optional) Fukui indices | xTB |
# | `TurbomoleSinglePointOptions` | DFT energy | Turbomole |
# | `TurbomoleFreehOptions` | thermochemistry (freeh) | Turbomole |
# | `TurbomoleFukuiOptions` | Fukui indices | Turbomole |
# | `TurbomoleNmrShieldingOptions` | NMR shieldings | Turbomole |
# | `TurbomoleVcdOptions` | VCD spectra | Turbomole |
# | `JaguarHydrogenAbstractionOptions` | H-abstraction energies | Jaguar |
# | `JaguarQmDescriptorsOptions` | QM descriptors | Jaguar |
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_calculate_properties.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** the xTB single-point calculator needs xTB. The Turbomole
# and Jaguar calculators need those respective programs.

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
# ## Imports and a 3D molecule
#
# Property calculations run on 3D conformers. We use methane as a small, fast
# test molecule: we embed a single conformer with RDKit and relax it with xTB so
# the geometry is sensible before we compute properties on it.

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.tasks import calculate_properties, conformer_generation, optimize

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

methane = Chem.MolFromSmiles("C")
methane.SetProp("_Name", "methane")
methane = conformer_generation.generate_rdkit_conformers(
    methane, method="ETKDGv3", max_conformers=1, scr=SCR
)

# Relax the geometry with xTB before calculating properties.
methane, _optimization_properties = optimize.optimize_mol(
    methane,
    config=[optimize.XtbOptions(solvation_model="alpb", solvent="water")],
    n_cores=1,
    scr=SCR,
    show_progress=True,
)
print(f"methane ready with {methane.GetNumConformers()} conformer")

# %% [markdown]
# ## Running property tasks on a single conformer
#
# Every calculator is invoked the same way through `calculate_property`, which
# operates on **one** conformer (selected by `conf_id`) and returns a dict
# mapping property names to `Property` objects (or `None` if the calculation
# failed). Read a value with `prop.get_property_value()`.
#
# Each task below gets its own cell that (1) builds its default configuration,
# (2) prints the resolved options, (3) runs the calculation on our methane
# conformer, and (4) reads the output. Each backend needs its external program
# (xTB, Turbomole or Jaguar) to be available in your software config.

# %% [markdown]
# ### xTB single point
#
# `XtbSinglePointOptions` runs a GFN2-xTB single point. Pick a solvation model
# (`alpb`/`gbsa`/`cosmo`) and solvent, or leave `solvation_model=None` for gas
# phase; set `calculate_fukui=True` to also get Fukui indices. Property names
# are auto-prefixed (e.g. `xtb_alpb_water_`).

# %%
xtb_sp_config = calculate_properties.XtbSinglePointOptions(solvation_model="alpb", solvent="water")
print("Configuration for xTB single point calculation:")
print(xtb_sp_config)

xtb_sp_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=xtb_sp_config, scr=SCR
)

# %%
print("Results of xTB single point calculation:")
for name, prop in xtb_sp_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# Below we show an example for running a g-xtb calculation. Your xtb installation needs to be recent to support this. At the time of writing, g-xtb has limited solvation support, and we are restricted to "gbe" and "cosmo" as solvation models

# %%
scr = Path(".").resolve()

xtb_sp_config = calculate_properties.XtbSinglePointOptions(
    solvation_model="cosmo", solvent="water", xtb_command_add=["--gxtb"]
)
print("Configuration for xTB single point calculation:")
print(xtb_sp_config)

gxtb_sp_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=xtb_sp_config, scr=scr, keep_files=True
)

print("Results of g-xtb single point calculation:")
for name, prop in gxtb_sp_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Turbomole single point
#
# `TurbomoleSinglePointOptions` runs a DFT single point (default BP86 /
# def2-TZVPD with COSMO) and returns the DFT energy. Requires Turbomole.

# %%
tm_sp_config = calculate_properties.TurbomoleSinglePointOptions()
print("Configuration for Turbomole ridft single point calculation:")
print(tm_sp_config)

tm_sp_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=tm_sp_config, scr=SCR
)

# %%
cosmo_file_as_str = calculate_properties.extract_cosmo_output(tm_sp_props, remove_from_dict=True)

print("Results of Turbomole ridft single point calculation:")
for name, prop in tm_sp_props.items():
    print(f"{name} = {prop.get_property_value()}")

print("\nCOSMO file content:")
print("\n".join(cosmo_file_as_str.split("\n")[:10]))  # Print first 10 lines of the COSMO file

# %% [markdown]
# ### Turbomole thermochemistry (freeh)
#
# `TurbomoleFreehOptions` runs a frequency calculation followed by `freeh` to
# obtain thermochemistry (HOMO/LUMO, entropy, enthalpy, ...), returned as
# several scalar properties. Requires Turbomole.

# %%
tm_freeh_config = calculate_properties.TurbomoleFreehOptions()
print("Configuration for Turbomole freeh calculation:")
print(tm_freeh_config)

tm_freeh_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=tm_freeh_config, scr=SCR
)

# %%
print("Results of Turbomole freeh calculation:")
for name, prop in tm_freeh_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Turbomole Fukui indices
#
# `TurbomoleFukuiOptions` computes condensed Fukui functions per atom plus
# global reactivity descriptors (ionization potential, hardness, ...). The
# per-atom values come back as `AtomBasedProperty` (one value per atom).
# Requires Turbomole.

# %%
tm_fukui_config = calculate_properties.TurbomoleFukuiOptions()
print("Configuration for Turbomole Fukui calculation:")
print(tm_fukui_config)

tm_fukui_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=tm_fukui_config, scr=SCR
)

# %%
print("Results of Turbomole Fukui calculation:")
for name, prop in tm_fukui_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Turbomole NMR shieldings
#
# `TurbomoleNmrShieldingOptions` computes isotropic NMR shieldings, returned as
# an `AtomBasedProperty` (one shielding per atom). Requires Turbomole.

# %%
tm_nmr_config = calculate_properties.TurbomoleNmrShieldingOptions()

print("Configuration for Turbomole NMR calculation:")
print(tm_nmr_config)

tm_nmr_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=tm_nmr_config, scr=SCR
)

# %%
print("Results of Turbomole NMR calculation:")
for name, prop in tm_nmr_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Turbomole VCD
#
# `TurbomoleVcdOptions` computes a vibrational circular dichroism spectrum,
# returned as a `TensorProperty`. Requires Turbomole.

# %%
tm_vcd_config = calculate_properties.TurbomoleVcdOptions()

print("Configuration for Turbomole VCD calculation:")
print(tm_vcd_config)

tm_vcd_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=tm_vcd_config, scr=SCR
)

# %%
print("Results of Turbomole VCD calculation:")
for name, prop in tm_vcd_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Jaguar hydrogen abstraction
#
# `JaguarHydrogenAbstractionOptions` computes per-hydrogen abstraction energies
# (by default abstracting from carbon), returned as an `AtomBasedProperty`.
# Requires Jaguar.

# %%
jaguar_habs_config = calculate_properties.JaguarHydrogenAbstractionOptions()

print("Configuration for Jaguar H-Abstraction energy calculation:")
print(jaguar_habs_config)

jaguar_habs_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=jaguar_habs_config, scr=SCR
)

# %% [markdown]
# In the above setting, we calculate an abstraction energy for any carbon atom in the molecule (due to the atom_idcs="C"). For our example compound, there is only one carbon atom to consider.

# %%
print("Results of Jaguar H-Abstraction energy calculation:")
for name, prop in jaguar_habs_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ### Jaguar QM descriptors
#
# `JaguarQmDescriptorsOptions` computes a set of QM descriptors (Fukui indices,
# Löwdin charges, NMR shifts) as `AtomBasedProperty` values. Requires Jaguar.

# %%
jaguar_desc_config = calculate_properties.JaguarQmDescriptorsOptions()
print("Configuration for Jaguar QM descriptors calculation:")
print(jaguar_desc_config)

jaguar_desc_props = calculate_properties.calculate_property(
    methane, conf_id=0, config=jaguar_desc_config, scr=SCR
)

# %%
print("Results of Jaguar QM descriptors calculation:")
for name, prop in jaguar_desc_props.items():
    print(f"{name} = {prop.get_property_value()}")

# %% [markdown]
# ## Parallelizing over conformers with `calculate_property_mol`
#
# `calculate_property_mol` runs a calculator on **every** conformer of a
# molecule and parallelizes across them when `n_cores > 1`. It returns one entry
# per conformer: a dict of `Property` objects on success, or `None` when that
# conformer failed or did not converge — the modern equivalent of filtering a
# result set down to the converged subset (cf. the old `extract_converged`
# helper).
#
# To make the error handling visible we give methane a second, deliberately
# broken conformer: we copy the relaxed geometry and move two hydrogens almost
# on top of each other. Turbomole cannot converge that distorted geometry, so
# its single point fails while the good conformer succeeds.
#

# %%
from rdkit.Geometry import Point3D

# Build a two-conformer methane: conformer 0 is the relaxed geometry from
# above, conformer 1 is a distorted copy in which two hydrogens are forced
# almost on top of each other. Turbomole cannot converge the distorted
# geometry, which illustrates the per-conformer error handling.
methane_2conf = Chem.Mol(methane)

base_conf = methane_2conf.GetConformers()[0]
distorted = Chem.Conformer(base_conf)
# Atoms 1-4 are the hydrogens; move H(2) to almost coincide with H(1).
h1_pos = distorted.GetAtomPosition(1)
distorted.SetAtomPosition(2, Point3D(h1_pos.x, h1_pos.y, h1_pos.z))
methane_2conf.AddConformer(distorted, assignId=True)

print(f"{methane_2conf.GetNumConformers()} conformers to evaluate")

# %%
tm_sp_config = calculate_properties.TurbomoleSinglePointOptions(use_cosmo=True, solvent="water")

# n_cores > 1 spreads the per-conformer single points across worker processes.
results = calculate_properties.calculate_property_mol(
    methane_2conf,
    config=tm_sp_config,
    n_cores=2,
    show_progress=True,
    scr=SCR,
)

# Split the results into converged conformers and failures. Failed /
# unconverged conformers are returned as None.
converged = [(i, props) for i, props in enumerate(results) if props is not None]
failed_indices = [i for i, props in enumerate(results) if props is None]

print(f"{len(converged)} converged, {len(failed_indices)} failed (indices {failed_indices})")
for conf_id, props in converged:
    values = {name: prop.get_property_value() for name, prop in props.items()}
    print(f"Conformer {conf_id}: {values}")

# %%
_tmpdir.cleanup()
