# %% [markdown]
# # Workflow: Constrained Optimization
#
# This notebook documents `qm_atlas.workflows.run_constrained_optimization`,
# which relaxes a single 3D geometry while keeping selected internal coordinates
# (by default torsions) restrained. It runs one or more optimization
# configurations and returns one optimized molecule per configuration — handy
# for probing how a structure responds to different constraint strengths.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook workflow_optimize_constrained.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** the default configuration uses Turbomole gradients driven
# by xTB (`XtbTurbomoleOptions`), so both xTB and Turbomole are needed. Supply a
# pure `XtbOptions` config to run with xTB only.

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
# ## Imports and a single-conformer input
#
# The workflow requires a molecule with **exactly one** conformer. We generate
# conformers with RDKit and keep the first.

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.tasks import conformer_generation, optimize
from qm_atlas.workflows import optimize_constrained

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

retinoic_acid_smi = "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1"  # CHEMBL38
mol = Chem.MolFromSmiles(retinoic_acid_smi)
mol.SetProp("_Name", "retinoic_acid")
retinoic_acid_single_conformation = conformer_generation.generate_rdkit_conformers(
    mol, method="ETKDGv3", max_conformers=1, rms_threshold=0.5, scr=SCR
)

print(f"Prepared molecule with {retinoic_acid_single_conformation.GetNumConformers()} conformer")

# %% [markdown]
# ## Run with the default configuration
#
# The default (`DEFAULT_OPTIMIZATION_CONFIG`) runs two constrained
# Turbomole/xTB optimizations with different force constants and returns one
# molecule per run. Each result carries an `Optimization_RMSD` property
# (`optimize_constrained.OPTIMIZATION_RMSD_PROP`) measuring how far the geometry
# moved.

# %%
# Uncomment to run (needs xTB + Turbomole):
# results = optimize_constrained.run_constrained_optimization(single, n_cores=1)
# for i, opt_mol in enumerate(results):
#     rmsd = opt_mol.GetProp(optimize_constrained.OPTIMIZATION_RMSD_PROP)
#     print(f"Run {i}: RMSD to input = {rmsd}")

# %% [markdown]
# ## Custom constraint configurations
#
# `optimization_config` is a list of *fallback chains*: each outer entry is run
# separately (producing one output molecule), and each inner list is a chain of
# configs tried in order until one succeeds. Here we run a single xTB once
# constrained and once unconstrained

# %%
xtb_only_config = [
    [
        optimize.XtbOptions(
            constrain=True,
            torsions_fixed=True,
            force_constant=1.0,
            solvation_model="alpb",
            solvent="water",
        ),
    ],
    [
        optimize.XtbOptions(
            constrain=False,
            solvation_model="alpb",
            solvent="water",
        ),
    ],
]

xtb_results = optimize_constrained.run_constrained_optimization(
    retinoic_acid_single_conformation, optimization_config=xtb_only_config, n_cores=1
)
print(f"Got {len(xtb_results)} optimized molecule(s)")

# %%
rmsd_constrained = Chem.rdMolAlign.GetBestRMS(retinoic_acid_single_conformation, xtb_results[0])
rmsd_unconstrained = Chem.rdMolAlign.GetBestRMS(retinoic_acid_single_conformation, xtb_results[1])

print(f"RMSD to input (constrained): {rmsd_constrained:.4f} Å")
print(f"RMSD to input (unconstrained): {rmsd_unconstrained:.4f} Å")

# %%
_tmpdir.cleanup()
