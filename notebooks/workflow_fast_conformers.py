# %% [markdown]
# # Workflow: Fast Conformers
#
# This notebook documents `qm_atlas.workflows.generate_fast_conformers`, a
# multi-stage pipeline that turns a graph into a representative, deduplicated
# set of xTB-optimized conformers:
#
# 1. Force-field / distance-geometry conformer generation.
# 2. Shape filtering + xTB geometry optimization.
# 3. Optional final re-optimization of the unique conformers.
#
# It is the workhorse used internally by the tautomer and scoring workflows.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook workflow_fast_conformers.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** xTB, plus a conformer generator (OMEGA or RDKit depending
# on your options).

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
# ## Imports and input molecule

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.workflows import fast_conformers

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

mol = Chem.MolFromSmiles("C[C@@H]([C@H](C)O)[C@H](N)C(=O)O")  # CHEMBL508094
mol.SetProp("_Name", "example")
mol

# %% [markdown]
# ## Run with defaults
#
# With no extra arguments the workflow uses `FAST_DEFAULT_OPTIONS` for conformer
# generation and `DEFAULT_OPTIMIZATION_CONFIG` (xTB) for optimization. It
# returns a single molecule carrying the surviving conformers.

# %%
result_mol = fast_conformers.generate_fast_conformers(
    mol,
    scr=SCR,
    n_cores=4,
    show_progress=True,
)
print(f"Fast-conformer workflow returned {result_mol.GetNumConformers()} conformers")

# %% [markdown]
# ## Customising the stages
#
# You can supply your own conformer-generation options and optimization configs.
# For example, use an RDKit generator and a laxer xTB optimization

# %%
from qm_atlas.tasks import conformer_generation, optimize

custom_gen = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        conformer_generation.RdkitOptions(method="KDG", max_conformers=50, rms_threshold=0.5),
    ],
    combination_name="fallback",
)

custom_mol = fast_conformers.generate_fast_conformers(
    mol,
    conformer_generation_options=custom_gen,
    optimization_config=[
        optimize.XtbOptions(solvation_model="alpb", solvent="water", opt_level="loose")
    ],
    scr=SCR,
    n_cores=4,
    show_progress=True,
)
print(f"Custom run returned {custom_mol.GetNumConformers()} conformers")

# %% [markdown]
# ## Adding a final optimization stage
#
# `final_optimization_config` re-optimizes the unique conformers at a higher
# level. Leave it empty (the default) to skip this step, or add e.g. a Turbomole
# DFT config (requires Turbomole):
#
# ```python
# fast_conformers.generate_fast_conformers(
#     mol,
#     final_optimization_config=[
#         optimize.XtbTurbomoleOptions(functional=["b-p"], basis="def2-TZVP"),
#     ],
#     scr=SCR,
# )
# ```

# %%
_tmpdir.cleanup()
