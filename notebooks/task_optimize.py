# %% [markdown]
# # Task: Geometry Optimization
#
# This notebook documents `qm_atlas.tasks.optimize`, which relaxes the 3D
# geometry of every conformer of a molecule using one of three drivers:
#
# | Driver | Options class | External program |
# | --- | --- | --- |
# | `xtb` | `XtbOptions` | xTB (fast, semi-empirical) |
# | `xtb_turbomole` | `XtbTurbomoleOptions` | Turbomole gradients + xTB stepper |
# | `jobex` | `JobexOptions` | Turbomole `jobex` (DFT) |
#
# You pass a **list** of option objects. The list acts as a *fallback chain*:
# the first driver is tried, and if it fails (or does not converge) the next one
# is used.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_optimize.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through the software config file pointed
# to by `QM_ATLAS_SOFTWARE_CONFIG_FILE`. Set it via a git-ignored `.env` file in
# the repository root (copy `.env.example` to `.env`); see the README
# (*Software Configuration*).
#
# **Prerequisites:** the `xtb` driver needs xTB. The `xtb_turbomole` and `jobex`
# drivers additionally need Turbomole.

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
# ## Imports and a starting geometry
#
# Optimization needs a molecule that already has 3D conformers. We generate a
# few with RDKit (see `task_conformer_generation` for details).

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.tasks import conformer_generation, optimize

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

mol = Chem.MolFromSmiles("C(C(=O)O)N")  # glycine
mol.SetProp("_Name", "glycine")
mol = conformer_generation.generate_rdkit_conformers(
    mol, method="KDG", max_conformers=2, rms_threshold=0.3, scr=SCR
)
print(f"Starting from {mol.GetNumConformers()} conformers")

# %% [markdown]
# ## Optimize with xTB (fast)
#
# `XtbOptions` defaults to GFN2-xTB (`method="2"`) with ALPB water solvation.
# Set `solvation_model=None` for gas phase.

# %%
xtb_config = optimize.XtbOptions(
    method="2",
    opt_level="normal",
    solvation_model="alpb",
    solvent="water",
    max_num_steps=200,
)

optimized_mol, per_conformer_properties = optimize.optimize_mol(
    mol,
    config=[xtb_config],
    n_cores=1,
    scr=SCR,
    show_progress=True,
)

print(f"Optimized {optimized_mol.GetNumConformers()} conformers")

# %% [markdown]
# ### Inspect the returned properties
#
# `optimize_mol` returns `(optimized_mol, per_conformer_properties)`, where the
# second element is one dict per conformer mapping a property name to a
# `Property` object. Read values with `.get_property_value()`.

# %%
for conf_id, props in enumerate(per_conformer_properties):
    if props is None:
        print(f"Conformer {conf_id}: optimization failed")
        continue
    values = {name: prop.get_property_value() for name, prop in props.items()}
    print(f"Conformer {conf_id}: {values}")

# %% [markdown]
# ## Pure DFT optimization with jobex
#
# `JobexOptions` drives Turbomole's `jobex` directly (all steps in Turbomole).

# %%
jobex_config = optimize.JobexOptions(
    functional=["b-p"],
    basis="def2-TZVP",
    use_cosmo=True,
    solvent="water",
    max_num_steps=2,
)

# Uncomment to run (needs Turbomole):
jobex_mol, jobex_props = optimize.optimize_mol(
    mol,
    config=[jobex_config],
    scr=SCR,
    n_cores=2,
    treat_unconverged_as_failure=False,
)

# %% [markdown]
# ## Letting xtb drive turbomole
#
# xtb can run turbomole's `rdgrad` and `ridft` and use it's own optimizer to drive the optimization based on that

# %%
xtb_turbomole_config = optimize.XtbTurbomoleOptions(
    functional=["b-p"],
    basis="def2-TZVP",
    use_cosmo=True,
    solvent="water",
    max_num_steps=5,
)

# Uncomment to run (needs Turbomole):
xtb_turbomole_mol, xtb_turbomole_props = optimize.optimize_mol(
    mol,
    config=[xtb_turbomole_config],
    scr=SCR,
    n_cores=2,
    treat_unconverged_as_failure=False,
)

# %% [markdown]
# ## Constrained optimization
#
# Both `XtbOptions` and `XtbTurbomoleOptions` support constraints. Set
# `constrain=True` and choose which internal coordinates to freeze
# (`torsions_fixed`, `angles_fixed`, `bonds_fixed`) plus a `force_constant`.
# See the `workflow_optimize_constrained` notebook for a higher-level entry
# point built on this.

# %%
constrained_config = optimize.XtbOptions(
    constrain=True,
    torsions_fixed=True,
    angles_fixed=False,
    bonds_fixed=False,
    force_constant=0.5,
)
constrained_config

# %% [markdown]
# ## Fallback chain: Several optimization attempts to try in order
#
# Passing several configs makes `optimize_mol` try them in order. Here a
# GFN2-xtb is attempted first optimization is tried first, falling
# back to GFN0-xtb it fails or does not converge. We artificially set the maximal
# number of steps low for the first run, such that the fallback is triggered.
#
# Let us first check that GFN2-xtb does indeed not converge in 2 steps:

# %%
fallback_config = [
    optimize.XtbOptions(
        method="2",
        opt_level="tight",
        max_num_steps=2,
        solvation_model="alpb",
        solvent="water",
    ),
]

opt_mol, opt_props = optimize.optimize_mol(
    mol,
    config=fallback_config,
    treat_unconverged_as_failure=True,
    scr=SCR,
    n_cores=1,
    keep_files=True,
)

# %%
fallback_config = [
    optimize.XtbOptions(
        method="2",
        opt_level="tight",
        max_num_steps=1,
        solvation_model="alpb",
        solvent="water",
    ),
    optimize.XtbOptions(
        method="0",
        opt_level="normal",
        max_num_steps=200,
        solvation_model="alpb",
        solvent="water",
    ),
]

# Uncomment to run (needs Turbomole):
opt_mol, opt_props = optimize.optimize_mol(
    mol, config=fallback_config, treat_unconverged_as_failure=True, scr=SCR, n_cores=2
)

# %%
_tmpdir.cleanup()
